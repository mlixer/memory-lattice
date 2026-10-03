// =============================================================================
// consolidator.js — Memory Pipeline: Cross-Session Consolidation (M2)
// =============================================================================
// Problem this solves: usage shifted toward many short ad-hoc chats, so a
// single topic fragments into shards across sessions. Retrieval pulls one
// shard; the rest stay buried. Neighbor expansion (reader-side) stitches
// memories back together TEMPORALLY; this module stitches them TOPICALLY.
//
// What it does, each run:
//   1. Scroll ALL base pipeline points (with vectors) out of the character's
//      Qdrant collection.
//   2. Cluster them by embedding similarity (greedy centroid grouping).
//   3. Keep clusters that are big enough AND span multiple sessions.
//   4. One LLM call per qualifying cluster → a single consolidated topic
//      memory that narrates the thread across time.
//   5. DELETE all previous consolidated points, embed + upsert the new ones.
//
// Design principles:
//   - ADDITIVE LAYER. Original summaries are never modified or deleted.
//     Consolidated points are a second kind of memory living alongside them.
//   - FULLY DERIVED, REBUILT EVERY RUN. Consolidations are regenerated from
//     scratch each time — no incremental merging, no staleness, no
//     migrations. Derived data you can always regenerate is data you never
//     have to repair.
//   - BEST-EFFORT. Any failure degrades: a failed cluster is skipped, a
//      failed run leaves the previous state standing. The pipeline never
//      blocks on consolidation.
//
// Payload contract for consolidated points:
//   {
//     text:         string          — the consolidated memory (injected verbatim)
//     isChunk:      true            — Qdrant Memory displays as Conversation(...)
//     speakers:     "User, <name>"
//     timestamp:    number          — ms of the NEWEST member (sorts at the
//                                     thread's latest activity)
//     kind:         "consolidated"  — the discriminator everything keys off
//     source:       "memPipeline"
//     dateFrom:     "YYYY-MM-DD"    — oldest member's session date
//     dateTo:       "YYYY-MM-DD"    — newest member's session date
//     memberCount:  number
//     sessionCount: number
//   }
//   NOTE: no sessionFile — this (plus the reader's guard) keeps consolidated
//   points out of neighbor-expansion anchoring, and out of wipeSessions'
//   selective deletes. The reader additionally excludes kind=consolidated
//   from temporal-neighbor pulls so they only ever surface as direct hits.
// =============================================================================

'use strict';

import { callLlm }                                  from './summarizer.js';
import { deriveCollectionName, embedText,
         DEFAULT_QDRANT_PORT, DEFAULT_EMBEDDING_PORT,
         DEFAULT_EMBEDDING_MODEL }                  from './qdrantIndexer.js';

const LOG_PREFIX = '[MemoryPipeline/Consolidate]';

// Hard ceiling on how many member summaries are fed to one LLM call. With
// well-chosen thresholds clusters stay small; this is protection against a
// runaway cluster blowing the context. The NEWEST members are kept.
const MAX_MEMBERS_PER_LLM_CALL = 20;

// =============================================================================
// Default prompt (editable in settings — Advanced drawer)
// =============================================================================

const DEFAULT_CONSOLIDATE_PROMPT = `You are weaving together related memories recorded across separate conversations. Each input memory is dated. Your task is to merge them into a single, cohesive memory of this one thread — how it began, how it evolved, and where it stood most recently.

Rules:

Ground everything in the input memories; do not invent details, events, or emotions that are not present in them.
Preserve specific language: where the memories quote the user's words, carry those quotes forward.
Note the passage of time naturally (for example, "over several weeks in March and April...").
When details conflict, the most recent memory reflects the current state; describe earlier details as how things were before.
Write in the first person, past tense, as one continuous memory.
Output plain prose only — no headers, no lists, no preamble, and do not mention that this is a summary or a merge.`;

// =============================================================================
// SECTION 1: Vector math (tiny, dependency-free)
// =============================================================================

function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
}

function normalize(v) {
    let n = 0;
    for (let i = 0; i < v.length; i++) n += v[i] * v[i];
    n = Math.sqrt(n);
    if (n === 0) return v.slice();
    return v.map(x => x / n);
}

// =============================================================================
// SECTION 2: Scroll all base points (with vectors)
// =============================================================================

/**
 * Pages through the collection and returns every base pipeline point —
 * source=memPipeline, NOT kind=consolidated — including vectors.
 *
 * @returns {Promise<object[]>} - [{ id, vector, payload }]
 */
async function scrollBasePoints(collectionName, qdrantPort = DEFAULT_QDRANT_PORT) {
    const url = `http://localhost:${qdrantPort}/collections/${collectionName}/points/scroll`;
    const points = [];
    let offset = null;

    do {
        const body = {
            filter: {
                must:     [{ key: 'source', match: { value: 'memPipeline' } }],
                must_not: [{ key: 'kind',   match: { value: 'consolidated' } }],
            },
            limit: 256,
            with_payload: true,
            with_vector: true,
        };
        if (offset !== null) body.offset = offset;

        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (!response.ok) {
            throw new Error(`${LOG_PREFIX} Scroll failed: HTTP ${response.status}`);
        }

        const data = await response.json();
        const batch = data?.result?.points ?? [];
        for (const pt of batch) {
            if (Array.isArray(pt.vector) && pt.payload?.text) {
                points.push(pt);
            }
        }
        offset = data?.result?.next_page_offset ?? null;
    } while (offset !== null);

    return points;
}

// =============================================================================
// SECTION 3: Clustering (greedy centroid grouping)
// =============================================================================

/**
 * Greedy single-pass clustering: walk points oldest-first; each point joins
 * the cluster whose centroid it's most similar to (if above threshold),
 * otherwise starts a new cluster. Centroids are running means, renormalised
 * on every add so cosine similarity stays meaningful.
 *
 * O(n * clusters) — trivial at hundreds of points, fine into the thousands.
 * Order-dependent, but oldest-first makes it deterministic per collection
 * state, and topic threads naturally accrete onto their earliest mention.
 *
 * @param {object[]} points    - [{ id, vector, payload }]
 * @param {number}   threshold - Cosine similarity required to join (0..1)
 * @returns {object[][]}       - Array of clusters (arrays of points)
 */
function clusterPoints(points, threshold) {
    // Oldest-first for determinism.
    const sorted = [...points].sort(
        (a, b) => (a.payload?.timestamp ?? 0) - (b.payload?.timestamp ?? 0));

    const clusters = []; // { sum: number[], centroid: number[], members: [] }

    for (const pt of sorted) {
        const v = normalize(pt.vector);

        let best = null;
        let bestSim = -1;
        for (const c of clusters) {
            const sim = dot(v, c.centroid);
            if (sim > bestSim) {
                bestSim = sim;
                best = c;
            }
        }

        if (best && bestSim >= threshold) {
            best.members.push(pt);
            for (let i = 0; i < best.sum.length; i++) best.sum[i] += v[i];
            best.centroid = normalize(best.sum);
        } else {
            clusters.push({ sum: v.slice(), centroid: v, members: [pt] });
        }
    }

    return clusters.map(c => c.members);
}

/**
 * A cluster qualifies for consolidation when it has at least `minCluster`
 * members spanning at least 2 distinct sessions. Same-session chunks
 * already received cross-chunk context from refinement — re-consolidating
 * a single session adds nothing.
 */
function qualifyingClusters(clusters, minCluster) {
    return clusters.filter(members => {
        if (members.length < minCluster) return false;
        const sessions = new Set(members.map(m => m.payload?.sessionFile).filter(Boolean));
        return sessions.size >= 2;
    });
}

// =============================================================================
// SECTION 4: Delete previous consolidated points
// =============================================================================

async function deleteConsolidatedPoints(collectionName, qdrantPort = DEFAULT_QDRANT_PORT) {
    const url = `http://localhost:${qdrantPort}/collections/${collectionName}/points/delete`;

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            filter: {
                must: [
                    { key: 'source', match: { value: 'memPipeline' } },
                    { key: 'kind',   match: { value: 'consolidated' } },
                ],
            },
        }),
    });

    if (!response.ok) {
        const msg = await response.text();
        throw new Error(`${LOG_PREFIX} Failed to delete previous consolidated points: ${msg}`);
    }

    console.log(`${LOG_PREFIX} Cleared previous consolidated points in "${collectionName}".`);
}

// =============================================================================
// SECTION 5: Consolidate one cluster (LLM call)
// =============================================================================

/**
 * Merges one cluster's member summaries into a single consolidated memory.
 *
 * @returns {Promise<string>} - Consolidated prose
 */
async function consolidateCluster(members, characterName, llmPort, promptOverride) {
    const systemPrompt = (typeof promptOverride === 'string' && promptOverride.trim().length > 0)
        ? promptOverride
        : DEFAULT_CONSOLIDATE_PROMPT;

    // Chronological. When a cluster exceeds the per-call cap, sample members
    // EVENLY across the timeline (always keeping the first and the last) so
    // the consolidated memory preserves the whole arc — how the thread began,
    // its middle, and where it stands — instead of only its newest stretch.
    const sorted = [...members].sort(
        (a, b) => (a.payload?.timestamp ?? 0) - (b.payload?.timestamp ?? 0));

    let trimmed = sorted;
    if (sorted.length > MAX_MEMBERS_PER_LLM_CALL) {
        trimmed = [];
        const step = (sorted.length - 1) / (MAX_MEMBERS_PER_LLM_CALL - 1);
        for (let i = 0; i < MAX_MEMBERS_PER_LLM_CALL; i++) {
            trimmed.push(sorted[Math.round(i * step)]);
        }
        console.warn(`${LOG_PREFIX} Cluster has ${sorted.length} members; sampling ${trimmed.length} evenly across ${sorted[0]?.payload?.sessionDate} → ${sorted[sorted.length - 1]?.payload?.sessionDate} for the LLM.`);
    }

    const inputBlock = trimmed
        .map(m => `[${m.payload?.sessionDate ?? 'unknown date'}]\n${m.payload.text}`)
        .join('\n\n---\n\n');

    const userPrompt = `The following ${trimmed.length} memories of conversations with ${characterName} are fragments of one recurring thread, in chronological order. Merge them as instructed in your system message.

--- MEMORIES BEGIN ---
${inputBlock}
--- MEMORIES END ---`;

    return await callLlm(systemPrompt, userPrompt, llmPort, 16384, false);
}

// =============================================================================
// SECTION 6: Orchestrator
// =============================================================================

/**
 * Full consolidation pass for one character. Safe to call after every
 * pipeline run; also callable standalone (manual rebuild button).
 *
 * Throws only on infrastructure failures (scroll/delete/upsert). Individual
 * cluster failures (LLM error, embedding error) are skipped with warnings.
 *
 * @param {string} characterName
 * @param {object} opts
 *   qdrantPort, embeddingPort, vectorSize, embeddingModel, llmPort,
 *   threshold      - cosine similarity to join a cluster (default 0.80)
 *   minCluster     - min members per qualifying cluster (default 3)
 *   maxClusters    - max clusters consolidated per run (default 12; largest first)
 *   promptOverride - custom consolidation prompt from settings
 * @returns {Promise<{basePoints:number, clustersQualified:number, indexed:number}>}
 */
async function runConsolidation(characterName, {
    qdrantPort     = DEFAULT_QDRANT_PORT,
    embeddingPort  = DEFAULT_EMBEDDING_PORT,
    embeddingModel = DEFAULT_EMBEDDING_MODEL,
    llmPort        = 8070,
    threshold      = 0.80,
    minCluster     = 3,
    maxClusters    = 12,
    promptOverride = null,
} = {}) {
    const collectionName = deriveCollectionName(characterName);
    console.log(`${LOG_PREFIX} Starting consolidation for "${characterName}" (threshold=${threshold}, minCluster=${minCluster}).`);

    // 1. Pull every base point with vectors.
    const basePoints = await scrollBasePoints(collectionName, qdrantPort);
    console.log(`${LOG_PREFIX} Scrolled ${basePoints.length} base point(s).`);

    if (basePoints.length < minCluster) {
        console.log(`${LOG_PREFIX} Not enough base points to consolidate. Done.`);
        return { basePoints: basePoints.length, clustersQualified: 0, indexed: 0 };
    }

    // 2-3. Cluster and filter.
    const clusters = clusterPoints(basePoints, threshold);
    let qualified = qualifyingClusters(clusters, minCluster);
    console.log(`${LOG_PREFIX} ${clusters.length} raw cluster(s) → ${qualified.length} qualifying (≥${minCluster} members, ≥2 sessions).`);

    // Largest clusters carry the most fragmentation — prioritise them.
    qualified.sort((a, b) => b.length - a.length);
    if (qualified.length > maxClusters) {
        console.log(`${LOG_PREFIX} Capping to the ${maxClusters} largest cluster(s) this run.`);
        qualified = qualified.slice(0, maxClusters);
    }

    // 4. Consolidate each cluster BEFORE deleting anything — if every LLM
    //    call fails we leave the previous consolidated layer standing.
    const speakers = `User, ${characterName}`;
    const newPoints = [];

    for (let i = 0; i < qualified.length; i++) {
        const members = qualified[i];
        const dates = members
            .map(m => m.payload?.sessionDate)
            .filter(Boolean)
            .sort();
        const timestamps = members
            .map(m => m.payload?.timestamp)
            .filter(t => typeof t === 'number');
        const sessions = new Set(members.map(m => m.payload?.sessionFile).filter(Boolean));

        console.log(`${LOG_PREFIX} Consolidating cluster ${i + 1}/${qualified.length} (${members.length} memories, ${sessions.size} sessions, ${dates[0]} → ${dates[dates.length - 1]})...`);

        let text;
        try {
            text = await consolidateCluster(members, characterName, llmPort, promptOverride);
        } catch (err) {
            console.warn(`${LOG_PREFIX} Cluster ${i + 1} LLM call failed: ${err.message}. Skipping cluster.`);
            continue;
        }

        let vector;
        try {
            vector = await embedText(text, embeddingPort, embeddingModel);
        } catch (err) {
            console.warn(`${LOG_PREFIX} Cluster ${i + 1} embedding failed: ${err.message}. Skipping cluster.`);
            continue;
        }

        newPoints.push({
            id: Date.now() + i,
            vector,
            payload: {
                // --- Qdrant Memory compatibility fields ---
                text,
                isChunk:   true,
                speakers,
                timestamp: timestamps.length ? Math.max(...timestamps) : Date.now(),

                // --- Consolidation metadata ---
                kind:         'consolidated',
                source:       'memPipeline',
                dateFrom:     dates[0] ?? null,
                dateTo:       dates[dates.length - 1] ?? null,
                memberCount:  members.length,
                sessionCount: sessions.size,
            },
        });
    }

    if (qualified.length > 0 && newPoints.length === 0) {
        console.warn(`${LOG_PREFIX} All ${qualified.length} cluster(s) failed. Previous consolidated layer left untouched.`);
        return { basePoints: basePoints.length, clustersQualified: qualified.length, indexed: 0 };
    }

    // 5. Rebuild the consolidated layer: delete old, insert new.
    await deleteConsolidatedPoints(collectionName, qdrantPort);

    if (newPoints.length > 0) {
        const upsertResponse = await fetch(`http://localhost:${qdrantPort}/collections/${collectionName}/points`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ points: newPoints }),
        });

        if (!upsertResponse.ok) {
            const msg = await upsertResponse.text();
            throw new Error(`${LOG_PREFIX} Upsert of consolidated points failed: ${msg}`);
        }
    }

    console.log(`${LOG_PREFIX} Consolidation complete: ${newPoints.length} consolidated memor${newPoints.length === 1 ? 'y' : 'ies'} indexed.`);
    return { basePoints: basePoints.length, clustersQualified: qualified.length, indexed: newPoints.length };
}

// =============================================================================
// Exports
// =============================================================================

export {
    runConsolidation,
    clusterPoints,
    qualifyingClusters,
    scrollBasePoints,
    deleteConsolidatedPoints,
    DEFAULT_CONSOLIDATE_PROMPT,
};
