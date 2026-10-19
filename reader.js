// =============================================================================
// reader.js — Memory Lattice: Read Path (per-message recall)
// =============================================================================
// Clean-room implementation, written from the published architecture
// write-up and this extension's own Qdrant schema (qdrantIndexer.js /
// consolidator.js). No third-party extension code was referenced.
//
// Responsibilities (per message):
//   1. Build a query from the outgoing message (+ optional recent context)
//   2. Embed it (same endpoint + model as the write path — one config)
//   3. Search the character's collection: top-k above a score threshold.
//      Consolidated topic memories are NOT filtered out — they compete
//      with base memories on relevance and blend into one list.
//   4. Temporal neighbor expansion: each retrieved BASE memory may bring
//      its nearest-in-time neighbors within a configurable window, so
//      recall arrives with its context attached. Consolidated points are
//      never anchors (they are derived, not moments).
//   5. Render chronologically and inject into the chat array at a
//      configurable depth from the end.
//
// Failure doctrine: recall is best-effort. ANY error degrades to no
// injection (the fact sheet still rides via the interceptor's own step)
// with one console warning. A memory outage must never block a message.
// =============================================================================

'use strict';

import { deriveCollectionName, embedText } from './qdrantIndexer.js';

const LOG_PREFIX = '[MemoryLattice:reader]';

// =============================================================================
// SECTION 0: Defaults
// =============================================================================

const DEFAULT_READER_SETTINGS = {
    recallEnabled:           true,
    recallK:                 3,     // memories retrieved per message
    recallThreshold:         0.4,   // min cosine similarity to inject at all
    recallPosition:          3,     // insertion depth: messages from the end
    recallRetainRecent:      0,     // prior messages joined into the query (0 = just the new message)
    recallNeighborWindowDays: 7,    // neighbor search window around a match (± days)
    recallNeighborsPerMatch: 2,
    recallNeighborsTotal:    4,
    // Connection overrides. Blank = http://localhost:<port> — which, since
    // these fetches run in the BROWSER, means the device browsing ST. A
    // remote/tailnet browser needs full URLs here (and those services must
    // be reachable + CORS-permitting from that origin).
    qdrantUrl:               '',
    embeddingUrl:            '',
    embeddingModel:          'nomic-embed-text',
    embeddingApi:            'ollama',   // 'ollama' | 'openai' (llama.cpp = openai)
    // The voice of recall — how memories introduce themselves. Editable in
    // the settings drawer; the intended path (as with every prompt here) is
    // that your companion eventually rewrites it in its own words.
    recallHeader: 'The following are your own memories of past conversations, recalled because they relate to this moment:',
};

// In-array marker: a private property stamped on the injected message so a
// second interceptor call in the same generation (double registration — a
// failure mode this house has actually hit) skips instead of double-pasting.
// Distinct from the fact sheet's marker in index.js, and from state-bridge's.
const RECALL_MARKER = '__lattice_recall';

/** Qdrant base URL: explicit setting wins; blank = loopback (browser-local). */
function qdrantBase(s) {
    return (s.qdrantUrl || `http://localhost:${s.qdrantPort}`).replace(/\/+$/, '');
}

// =============================================================================
// SECTION 1: Qdrant read helpers
// =============================================================================

/**
 * Vector search over a character's collection. Blended: consolidated
 * topic memories and base memories compete on equal footing.
 *
 * @returns {Promise<object[]>} - [{ id, score, payload }]
 */
async function searchMemories(collectionName, vector, k, scoreThreshold, base) {
    const response = await fetch(`${base}/collections/${collectionName}/points/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            vector,
            limit: k,
            score_threshold: scoreThreshold,
            with_payload: true,
        }),
    });

    if (!response.ok) {
        // 404 means the collection doesn't exist yet (pipeline never ran
        // for this character) — that's a normal state, not an error.
        if (response.status === 404) return [];
        throw new Error(`Qdrant search error: HTTP ${response.status}`);
    }

    const data = await response.json();
    return Array.isArray(data?.result) ? data.result : [];
}

/**
 * Temporal neighbors of one base memory: points whose timestamp falls
 * within ±windowMs of the anchor, nearest first. Consolidated points are
 * excluded by filter; the anchor and anything already collected are
 * excluded by the caller.
 *
 * Qdrant can't order a scroll by distance-to-a-timestamp, so we pull the
 * window (bounded) and sort client-side — at a few hundred points per
 * collection this is effectively free.
 */
async function fetchNeighbors(collectionName, anchorTimestamp, windowMs, base) {
    const response = await fetch(`${base}/collections/${collectionName}/points/scroll`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            filter: {
                must: [{
                    key: 'timestamp',
                    range: { gte: anchorTimestamp - windowMs, lte: anchorTimestamp + windowMs },
                }],
                must_not: [{
                    key: 'kind',
                    match: { value: 'consolidated' },
                }],
            },
            // 64 covers the whole window at summary-memory density (a few
            // points/day). In a far denser corpus the window could overflow
            // this cap, and "nearest first" would sort only what arrived.
            limit: 64,
            with_payload: true,
        }),
    });

    if (!response.ok) return []; // neighbors are a bonus, never a blocker

    const data = await response.json();
    const points = Array.isArray(data?.result?.points) ? data.result.points : [];

    return points.sort((a, b) =>
        Math.abs((a.payload?.timestamp ?? 0) - anchorTimestamp) -
        Math.abs((b.payload?.timestamp ?? 0) - anchorTimestamp));
}

/**
 * Exact point count for a collection — the "Indexed Memories" display.
 * Returns null when the collection doesn't exist yet.
 */
async function countMemories(characterName, qdrantPort, qdrantUrl = '') {
    const collectionName = deriveCollectionName(characterName);
    const base = (qdrantUrl || `http://localhost:${qdrantPort}`).replace(/\/+$/, '');
    const response = await fetch(`${base}/collections/${collectionName}/points/count`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ exact: true }),
    });
    if (!response.ok) return null;
    const data = await response.json();
    return Number.isFinite(data?.result?.count) ? data.result.count : null;
}

// =============================================================================
// SECTION 2: Connection tests (settings-drawer buttons)
// =============================================================================

/** @returns {Promise<{ok: boolean, detail: string}>} */
async function testQdrantConnection(qdrantPort, qdrantUrl = '') {
    const base = (qdrantUrl || `http://localhost:${qdrantPort}`).replace(/\/+$/, '');
    try {
        const response = await fetch(`${base}/collections`, { method: 'GET' });
        if (!response.ok) return { ok: false, detail: `Qdrant answered HTTP ${response.status}` };
        const data = await response.json();
        const n = data?.result?.collections?.length ?? 0;
        return { ok: true, detail: `Qdrant reachable — ${n} collection(s).` };
    } catch (err) {
        return { ok: false, detail: `No response from Qdrant at port ${qdrantPort} (${err.message}). Is it running? If the console shows CORS with a real status code, the server needs its allowed-origins set.` };
    }
}

/** @returns {Promise<{ok: boolean, detail: string}>} */
async function testEmbeddingConnection(embeddingPort, embeddingModel, embeddingUrl = '', embeddingApi = 'ollama') {
    try {
        const vector = await embedText('connection test', embeddingPort, embeddingModel, embeddingUrl, embeddingApi);
        return { ok: true, detail: `Embedding endpoint reachable — returned a ${vector.length}-dim vector.` };
    } catch (err) {
        return { ok: false, detail: `Embedding call failed at port ${embeddingPort} (${err.message}). Check the service is running, the model is pulled, and allowed-origins (e.g. OLLAMA_ORIGINS) is set.` };
    }
}

// =============================================================================
// SECTION 3: Query construction
// =============================================================================

/**
 * Builds the retrieval query from the chat array ST hands the interceptor.
 * Base: the latest user message. With retainRecent > 0, that many
 * immediately-preceding messages (either speaker) are prepended, giving
 * short follow-ups ("and then what?") the context they lean on.
 */
function buildQueryText(chat, retainRecent) {
    if (!Array.isArray(chat) || chat.length === 0) return '';

    // Find the last real user message (skip injected system entries).
    let lastUserIdx = -1;
    for (let i = chat.length - 1; i >= 0; i--) {
        if (chat[i]?.is_user && !chat[i]?.is_system) { lastUserIdx = i; break; }
    }
    if (lastUserIdx === -1) return '';

    const parts = [];
    const from = Math.max(0, lastUserIdx - Math.max(0, retainRecent));
    for (let i = from; i <= lastUserIdx; i++) {
        const m = chat[i];
        if (m?.is_system || typeof m?.mes !== 'string') continue;
        const text = m.mes.trim();
        if (text.length > 0) parts.push(text);
    }
    return parts.join('\n');
}

// =============================================================================
// SECTION 4: Retrieval core
// =============================================================================

/**
 * The full read path for one query. Pure (no chat mutation) so the
 * Preview button and the interceptor share this exactly.
 *
 * @returns {Promise<{items: object[], queryText: string}>}
 *   items: [{ id, text, timestamp, kind, score|null, isNeighbor }],
 *   chronological. Empty array when nothing clears the threshold.
 */
async function retrieveMemories(queryText, characterName, s) {
    if (!queryText || !characterName) return { items: [], queryText };

    const collectionName = deriveCollectionName(characterName);
    const base = qdrantBase(s);
    const vector = await embedText(
        queryText, s.embeddingPort, s.embeddingModel, s.embeddingUrl, s.embeddingApi);

    const hits = await searchMemories(
        collectionName, vector, s.recallK, s.recallThreshold, base);
    if (hits.length === 0) return { items: [], queryText };

    const seen  = new Set(hits.map(h => h.id));
    const items = hits.map(h => ({
        id:         h.id,
        text:       h.payload?.text ?? '',
        timestamp:  h.payload?.timestamp ?? 0,
        kind:       h.payload?.kind ?? 'memory',
        dateFrom:   h.payload?.dateFrom ?? null,
        dateTo:     h.payload?.dateTo ?? null,
        score:      h.score,
        isNeighbor: false,
    }));

    // --- Temporal neighbor expansion (base memories only) -------------------
    const windowMs = Math.max(0, s.recallNeighborWindowDays) * 24 * 60 * 60 * 1000;
    let neighborBudget = Math.max(0, s.recallNeighborsTotal);

    if (windowMs > 0 && neighborBudget > 0) {
        for (const hit of hits) {
            if (neighborBudget <= 0) break;
            if (hit.payload?.kind === 'consolidated') continue;        // never an anchor
            const anchorTs = hit.payload?.timestamp;
            if (!Number.isFinite(anchorTs)) continue;

            const candidates = await fetchNeighbors(
                collectionName, anchorTs, windowMs, base);

            let taken = 0;
            for (const c of candidates) {
                if (taken >= s.recallNeighborsPerMatch || neighborBudget <= 0) break;
                if (seen.has(c.id)) continue;
                seen.add(c.id);
                items.push({
                    id:         c.id,
                    text:       c.payload?.text ?? '',
                    timestamp:  c.payload?.timestamp ?? 0,
                    // Neighbors can be shelf entries (kind 'hands'/'gift') as
                    // well as chat memories — carry the real kind through.
                    kind:       c.payload?.kind ?? 'memory',
                    dateFrom:   null,
                    dateTo:     null,
                    score:      null,   // neighbors ride on adjacency, not similarity
                    isNeighbor: true,
                });
                taken++;
                neighborBudget--;
            }
        }
    }

    // Chronological order: recall reads as a timeline, not a ranking.
    items.sort((a, b) => a.timestamp - b.timestamp);
    return { items, queryText };
}

// =============================================================================
// SECTION 5: Rendering
// =============================================================================

/** Human-scale age: "today", "yesterday", "5d ago", "3w ago", "4mo ago", "1y ago". */
function relativeAge(ts) {
    if (!Number.isFinite(ts) || ts <= 0) return '';
    const days = Math.floor((Date.now() - ts) / 86400000);
    if (days < 0) return '';
    if (days === 0) return 'today';
    if (days === 1) return 'yesterday';
    if (days < 14) return `${days}d ago`;
    if (days < 60) return `${Math.round(days / 7)}w ago`;
    if (days < 540) return `${Math.round(days / 30)}mo ago`;
    return `${(days / 365).toFixed(1).replace(/\.0$/, '')}y ago`;
}

function formatItemDate(item) {
    const age = relativeAge(item.timestamp);
    const suffix = age ? ` · ${age}` : '';
    if (item.kind === 'consolidated' && item.dateFrom && item.dateTo) {
        return (item.dateFrom === item.dateTo
            ? item.dateFrom
            : `${item.dateFrom} → ${item.dateTo}`) + suffix;
    }
    if (Number.isFinite(item.timestamp) && item.timestamp > 0) {
        return new Date(item.timestamp).toISOString().split('T')[0] + suffix;
    }
    return 'undated';
}

/** Renders retrieved items into the single injected block. */
function renderMemoriesBlock(items, header) {
    if (!items || items.length === 0) return '';
    const lines = items.map(item => `(${formatItemDate(item)}) ${item.text}`.trim());
    return `${header}\n\n${lines.join('\n\n')}`;
}

/** Rough token estimate for budget awareness (chars ÷ 3.5). */
function estimateTokens(text) {
    return Math.ceil((text?.length ?? 0) / 3.5);
}

// =============================================================================
// SECTION 6: Injection (called from the extension's generate_interceptor)
// =============================================================================

// Swipe/regen cache: the same outgoing message should not re-run retrieval
// (same query → same memories; re-rolling the dice on recall would make
// swipes incomparable). Keyed by character + query text.
let lastInjection = { key: null, block: null };

/**
 * Retrieves and splices the recall block into the chat array.
 * Mutates `chat` in place, like the fact-sheet injection does.
 *
 * @param {object[]} chat           - ST's chat array (newest last)
 * @param {string}   characterName
 * @param {object}   s              - full extension settings
 * @param {string}   type           - ST generation type ('quiet', 'swipe', ...)
 * @returns {Promise<number>}       - tokens injected (0 = nothing injected)
 */
async function injectRecalledMemories(chat, characterName, s, type) {
    if (!s.recallEnabled) return 0;
    if (type === 'quiet') return 0;   // internal/system generations don't need recall
    // Double-call guard: a second interceptor invocation in the same
    // generation must not paste the block twice (the cache below dedupes
    // retrieval, not the splice).
    if (chat.some(m => m && m[RECALL_MARKER])) return 0;

    const queryText = buildQueryText(chat, s.recallRetainRecent);
    if (!queryText) return 0;

    const cacheKey = `${characterName}|${queryText}`;
    let block;

    if (lastInjection.key === cacheKey && lastInjection.block) {
        block = lastInjection.block;  // swipe/regen of the same message
    } else {
        const { items } = await retrieveMemories(queryText, characterName, s);
        if (items.length === 0) {
            lastInjection = { key: cacheKey, block: null };
            return 0;
        }
        block = renderMemoriesBlock(items, s.recallHeader);
        lastInjection = { key: cacheKey, block };
    }

    if (!block) return 0;

    // Insert `recallPosition` messages from the end (clamped), as a system
    // entry — the same shape the fact-sheet injection uses.
    const depth = Math.max(0, Math.min(chat.length, s.recallPosition));
    const at = chat.length - depth;
    chat.splice(at, 0, {
        is_user:   false,
        is_system: true,
        name:      'System',
        mes:       block,
        send_date: new Date().toISOString(),
        [RECALL_MARKER]: true,
    });
    console.log(`${LOG_PREFIX} recall spliced at index ${at} of ${chat.length - 1} (depth ${depth} from end).`);

    return estimateTokens(block);
}

// =============================================================================
// SECTION 7: Preview (settings-drawer test flow, no message sent)
// =============================================================================

/**
 * Runs the full read path for an arbitrary query and returns a report:
 * what would be injected, each item's score, and the token cost. This is
 * the tuning instrument — thresholds and counts should be set from what
 * this shows, not from guesses.
 */
async function previewRecall(queryText, characterName, s, wide = false) {
    // Wide mode: the tuning instrument's zoom-out — top 20 by similarity with
    // NO threshold, so you can see where a known-relevant memory actually
    // ranks before blaming the threshold, k, or the embedding model.
    const eff = wide
        ? { ...s, recallK: Math.max(20, s.recallK), recallThreshold: 0,
            recallNeighborsTotal: 0 }   // raw ranking, no neighbor noise
        : s;
    const { items } = await retrieveMemories(queryText, characterName, eff);
    if (wide) items.sort((a, b) => (b.score ?? -1) - (a.score ?? -1));

    if (items.length === 0) {
        return { report: wide
            ? 'Nothing retrieved at all — collection empty or unreachable.'
            : `No memories cleared the threshold (${s.recallThreshold}) for this query.`, tokens: 0 };
    }

    const lines = items.map(item => {
        // Kind first (ARC / HANDS / GIFT / MEMORY), neighbor flag appended —
        // shelf entries (the hands' own writing) are labeled honestly.
        const KIND_TAGS = { consolidated: 'ARC', hands: 'HANDS', gift: 'GIFT' };
        let tag = KIND_TAGS[item.kind] ?? 'MEMORY';
        if (item.isNeighbor) tag += '+NBR';
        const score = item.score !== null ? item.score.toFixed(3) : '  --  ';
        const below = wide && item.score !== null && item.score < s.recallThreshold
            ? ' | BELOW threshold' : '';
        return `[${tag.padEnd(8)} | score ${score}${below} | ${formatItemDate(item)}]\n${item.text}`;
    });

    const block  = renderMemoriesBlock(items, s.recallHeader);
    const tokens = estimateTokens(block);

    const report = wide
        ? `WIDE view — top ${items.length} by similarity, threshold ignored `
          + `(your threshold: ${s.recallThreshold}, your k: ${s.recallK}):\n\n` + lines.join('\n\n')
        : `${items.length} item(s) would be injected (~${tokens} tokens):\n\n` + lines.join('\n\n');

    return { report, tokens };
}

// =============================================================================
// Exports
// =============================================================================

export {
    DEFAULT_READER_SETTINGS,
    retrieveMemories,
    renderMemoriesBlock,
    injectRecalledMemories,
    previewRecall,
    countMemories,
    testQdrantConnection,
    testEmbeddingConnection,
    buildQueryText,
    estimateTokens,
};
