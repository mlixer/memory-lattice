// =============================================================================
// qdrantIndexer.js — Memory Pipeline: Qdrant Wipe + Reindex
// =============================================================================
// Responsibilities:
//   1. Ensure a per-character collection exists in Qdrant
//   2. Wipe all existing vectors for the character (full wipe strategy)
//   3. Generate embeddings for each summary via the embedding endpoint
//   4. Insert fresh vectors with rich metadata payload into Qdrant
//
// Qdrant API used:
//   GET  /collections/{name}                   — check if collection exists
//   PUT  /collections/{name}                   — create collection
//   POST /collections/{name}/points/delete     — wipe by character filter
//   PUT  /collections/{name}/points            — upsert new points
//
// Embedding API (OpenAI-compatible, e.g. Ollama or llama.cpp):
//   POST /v1/embeddings
// =============================================================================

'use strict';

const LOG_PREFIX = '[MemoryPipeline/Qdrant]';

// Collection name prefix — matches the default in the Qdrant Memory
// extension (settings.collectionName = "mem"), so its generate_interceptor
// finds our indexed summaries on retrieval.
const COLLECTION_PREFIX = 'mem_';

// Default ports — both configurable in extension settings.
const DEFAULT_QDRANT_PORT    = 6333;
const DEFAULT_EMBEDDING_PORT = 11434; // Ollama default

// Default embedding model name. Ollama's /api/embeddings requires the model
// field; llama.cpp / vllm ignore it. Override in settings if needed.
const DEFAULT_EMBEDDING_MODEL = 'nomic-embed-text';

// Default vector size — must match the embedding model's output dimensions.
// 768 = nomic-embed-text. Must match whatever the Qdrant Memory extension
// uses, since both write to and read from the same collection.
const DEFAULT_VECTOR_SIZE = 768;


// =============================================================================
// SECTION 1: Collection Name
// =============================================================================

/**
 * Derives a safe Qdrant collection name from a character's NAME.
 *
 * Matches the Qdrant Memory extension's getCollectionName() exactly so its
 * generate_interceptor will retrieve from the same collection we write to:
 *   - lowercase
 *   - replace any char outside [a-z0-9_-] with underscore
 *   - collapse repeated underscores
 *   - trim leading/trailing underscores
 *   - prefix with "mem_"
 *
 * Input:  "Fin"            → Output: "mem_fin"
 * Input:  "Dr. Smith"      → Output: "mem_dr_smith"
 * Input:  "My Char (v2)"   → Output: "mem_my_char_v2"
 *
 * @param {string} characterName  - Display name (NOT avatar filename)
 * @returns {string}
 */
function deriveCollectionName(characterName) {
    const sanitized = characterName
        .toLowerCase()
        .replace(/[^a-z0-9_-]/g, '_')
        .replace(/_+/g, '_')
        .replace(/^_|_$/g, '');
    return `${COLLECTION_PREFIX}${sanitized}`;
}


// =============================================================================
// SECTION 2: Embedding
// =============================================================================

/**
 * Generates a vector embedding for a text string via Ollama's native
 * /api/embeddings endpoint. This matches the working setup the user's
 * Qdrant Memory extension uses (custom local endpoint, Ollama on 11434).
 *
 * Ollama request:  { model: string, prompt: string }
 * Ollama response: { embedding: number[] }
 *
 * When swapping to llama.cpp / vllm later, switch this URL to /v1/embeddings
 * and change request/response handling to the OpenAI-compatible format:
 *   request:  { model, input }
 *   response: { data: [{ embedding }] }
 *
 * @param {string} text            - The summary text to embed
 * @param {number} embeddingPort   - Port of the embedding service
 * @param {string} embeddingModel  - Model name (required by Ollama)
 * @returns {Promise<number[]>}    - The embedding vector
 */
async function embedText(text, embeddingPort = DEFAULT_EMBEDDING_PORT, embeddingModel = DEFAULT_EMBEDDING_MODEL) {
    const url = `http://localhost:${embeddingPort}/api/embeddings`;

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model: embeddingModel,
            prompt: text,
        }),
    });

    if (!response.ok) {
        throw new Error(`${LOG_PREFIX} Embedding API error: HTTP ${response.status}`);
    }

    const data = await response.json();
    const vector = data?.embedding;

    if (!Array.isArray(vector) || vector.length === 0) {
        throw new Error(`${LOG_PREFIX} Embedding API returned empty or invalid vector.`);
    }

    return vector;
}


// =============================================================================
// SECTION 3: Collection Management
// =============================================================================

/**
 * Checks if a Qdrant collection exists for this character.
 * Creates it with the correct vector configuration if it doesn't.
 *
 * Why Cosine distance?
 * Semantic text embeddings are unit-normalised — cosine similarity is the
 * standard and most accurate distance metric for this use case.
 *
 * @param {string} collectionName
 * @param {number} vectorSize      - Must match the embedding model's dimensions
 * @param {number} qdrantPort
 */
async function ensureCollection(collectionName, vectorSize = DEFAULT_VECTOR_SIZE, qdrantPort = DEFAULT_QDRANT_PORT) {
    const baseUrl = `http://localhost:${qdrantPort}`;

    // Check if collection already exists.
    const checkResponse = await fetch(`${baseUrl}/collections/${collectionName}`);

    if (checkResponse.ok) {
        console.log(`${LOG_PREFIX} Collection "${collectionName}" already exists.`);
        return;
    }

    if (checkResponse.status !== 404) {
        throw new Error(`${LOG_PREFIX} Unexpected response checking collection: HTTP ${checkResponse.status}`);
    }

    // Collection doesn't exist — create it.
    console.log(`${LOG_PREFIX} Creating collection "${collectionName}" (vector size: ${vectorSize})...`);

    const createResponse = await fetch(`${baseUrl}/collections/${collectionName}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            vectors: {
                size: vectorSize,
                distance: 'Cosine',
            },
        }),
    });

    if (!createResponse.ok) {
        const msg = await createResponse.text();
        throw new Error(`${LOG_PREFIX} Failed to create collection "${collectionName}": ${msg}`);
    }

    console.log(`${LOG_PREFIX} Collection "${collectionName}" created.`);
}


// =============================================================================
// SECTION 4: Wipe
// =============================================================================

/**
 * Deletes ALL points in a collection. This is the "full wipe" strategy:
 * every pipeline run starts clean.
 *
 * Why wipe the whole collection (not just our entries)?
 * Each collection is per-character (mem_fin, mem_aria, ...). The Qdrant
 * Memory extension may have auto-saved raw chat chunks into this same
 * collection before we took over. Wiping everything ensures only our
 * curated summaries remain after the pipeline runs.
 *
 * The user should also disable Qdrant Memory's autoSaveMemories setting
 * so it doesn't keep adding noisy raw chunks alongside our summaries.
 *
 * @param {string} collectionName
 * @param {number} qdrantPort
 */
async function wipeCollection(collectionName, qdrantPort = DEFAULT_QDRANT_PORT) {
    const url = `http://localhost:${qdrantPort}/collections/${collectionName}/points/delete`;

    // An empty filter matches every point in the collection.
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            filter: { must: [] },
        }),
    });

    if (!response.ok) {
        const msg = await response.text();
        throw new Error(`${LOG_PREFIX} Failed to wipe collection "${collectionName}": ${msg}`);
    }

    console.log(`${LOG_PREFIX} Wiped all points from collection "${collectionName}".`);
}

/**
 * Deletes only the points belonging to the given session files — the
 * "selective wipe" strategy used on incremental runs.
 *
 * Why this exists:
 * The pipeline only reprocesses chat sessions modified since the last run
 * (filtered by last_indexed_ms). But a full wipeCollection() would destroy
 * vectors for ALL sessions, including dormant ones that won't be reprocessed
 * and therefore won't be re-inserted — silently losing their memories.
 * Selective wipe removes only the sessions we're about to re-index, leaving
 * dormant sessions' vectors untouched.
 *
 * We scope the delete to our own entries (source == "memPipeline") as well,
 * so we never disturb anything another writer may have placed in the
 * collection. Every point we index carries both sessionFile and source.
 *
 * @param {string}   collectionName
 * @param {string[]} sessionFiles   - Distinct source chat files to clear
 * @param {number}   qdrantPort
 */
async function wipeSessions(collectionName, sessionFiles, qdrantPort = DEFAULT_QDRANT_PORT) {
    if (!Array.isArray(sessionFiles) || sessionFiles.length === 0) {
        return;
    }

    const url = `http://localhost:${qdrantPort}/collections/${collectionName}/points/delete`;

    // MatchAny ("any": [...]) deletes points whose sessionFile is any of the
    // updated files; the source clause restricts to our own entries.
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            filter: {
                must: [
                    { key: 'sessionFile', match: { any: sessionFiles } },
                    { key: 'source',      match: { value: 'memPipeline' } },
                ],
            },
        }),
    });

    if (!response.ok) {
        const msg = await response.text();
        throw new Error(`${LOG_PREFIX} Failed to wipe ${sessionFiles.length} session(s) in "${collectionName}": ${msg}`);
    }

    console.log(`${LOG_PREFIX} Selectively wiped points for ${sessionFiles.length} updated session(s) in "${collectionName}".`);
}


// =============================================================================
// SECTION 5: Indexing
// =============================================================================

/**
 * Generates embeddings for all summaries and upserts them into Qdrant.
 *
 * Payload schema is kept compatible with the Qdrant Memory extension's
 * retrieval/formatting logic, so its generate_interceptor can inject our
 * summaries into chat without any changes on its side. Required fields:
 *
 *   {
 *     text:        string   — REQUIRED. Qdrant Memory injects this verbatim.
 *     isChunk:     true     — Causes display as "Conversation (speakers)"
 *     speakers:    string   — Comma-separated participant labels
 *     timestamp:   number   — Unix ms timestamp of the session
 *   }
 *
 * Extra fields we add for our own debugging / future use (Qdrant Memory
 * ignores anything beyond the four above):
 *
 *   {
 *     sessionFile: string   — Source chat file
 *     sessionDate: string   — YYYY-MM-DD
 *     chunkIndex:  number   — Position within the session
 *     source:      "memPipeline"  — So we can tell our entries apart later
 *   }
 *
 * We batch upserts in groups of 100 to avoid oversized requests.
 * Embeddings are generated sequentially to avoid overwhelming Ollama.
 *
 * @param {string}   characterName  - Display name; used for collection naming
 *                                    AND speaker labels (must match what the
 *                                    Qdrant Memory extension sees on retrieval)
 * @param {object[]} summaries      - From runPipeline(): { sessionFile, chunkIndex, sessionDate, text }
 * @param {number}   qdrantPort
 * @param {number}   embeddingPort
 * @param {number}   vectorSize     - Must match embedding model dimensions
 * @param {string}   embeddingModel - Embedding model name (Ollama requires this)
 * @param {boolean}  fullWipe       - When true, wipe the WHOLE collection before
 *                                    indexing (correct only when summaries cover
 *                                    every session, i.e. a first run or forced
 *                                    full rebuild where since==0). When false
 *                                    (incremental run), only the sessions present
 *                                    in `summaries` are wiped and replaced, so
 *                                    dormant sessions' memories survive.
 * @returns {Promise<number>}       - Count of vectors indexed
 */
async function indexSummaries(
    characterName,
    summaries,
    qdrantPort      = DEFAULT_QDRANT_PORT,
    embeddingPort   = DEFAULT_EMBEDDING_PORT,
    vectorSize      = DEFAULT_VECTOR_SIZE,
    embeddingModel  = DEFAULT_EMBEDDING_MODEL,
    fullWipe        = true,
) {
    if (!summaries || summaries.length === 0) {
        console.log(`${LOG_PREFIX} No summaries to index for "${characterName}".`);
        return 0;
    }

    const collectionName = deriveCollectionName(characterName);

    // Ensure the collection exists before trying to write to it.
    await ensureCollection(collectionName, vectorSize, qdrantPort);

    if (fullWipe) {
        // Full rebuild (since==0): summaries cover every session, so a clean
        // slate is correct and also clears any orphaned/foreign points.
        await wipeCollection(collectionName, qdrantPort);
    } else {
        // Incremental run: only clear the sessions we're about to re-insert.
        // Dormant sessions that weren't reprocessed keep their vectors. This
        // is the fix for the data-loss bug where a whole-collection wipe
        // destroyed memories for sessions that wouldn't be re-indexed.
        const updatedSessions = [...new Set(summaries.map(s => s.sessionFile))];
        await wipeSessions(collectionName, updatedSessions, qdrantPort);
    }

    // Build Qdrant point objects.
    const points = [];
    const speakers = `User, ${characterName}`;

    for (let i = 0; i < summaries.length; i++) {
        const summary = summaries[i];

        console.log(`${LOG_PREFIX} Embedding summary ${i + 1}/${summaries.length} (session: ${summary.sessionDate})...`);

        let vector;
        try {
            vector = await embedText(summary.text, embeddingPort, embeddingModel);
        } catch (err) {
            // A single failed embedding should not abort the whole index run.
            console.warn(`${LOG_PREFIX} Skipping summary ${i + 1} — embedding failed: ${err.message}`);
            continue;
        }

        // Qdrant point IDs must be unsigned integers or UUIDs.
        // Date.now() + i is unique within a run; sequential embeds advance the
        // clock more than enough between iterations to avoid collisions.
        const id = Date.now() + i;

        points.push({
            id,
            vector,
            payload: {
                // --- Qdrant Memory compatibility fields ---
                text:        summary.text,
                isChunk:     true,
                speakers,
                // M4: prefer the chunk's own precise timestamp (from message
                // send_date) when the pipeline provides one; day-granularity
                // sessionDate remains the fallback for older callers.
                timestamp:   (typeof summary.timestampMs === 'number' && Number.isFinite(summary.timestampMs))
                                 ? summary.timestampMs
                                 : new Date(summary.sessionDate).getTime(),

                // --- Our own metadata ---
                sessionFile: summary.sessionFile,
                sessionDate: summary.sessionDate,
                chunkIndex:  summary.chunkIndex,
                source:      'memPipeline',
            },
        });
    }

    if (points.length === 0) {
        console.warn(`${LOG_PREFIX} All embeddings failed for "${characterName}". Nothing indexed.`);
        return 0;
    }

    // Upsert in batches of 100 to keep request payloads manageable.
    const BATCH_SIZE = 100;
    const baseUrl = `http://localhost:${qdrantPort}`;

    for (let b = 0; b < points.length; b += BATCH_SIZE) {
        const batch = points.slice(b, b + BATCH_SIZE);

        const upsertResponse = await fetch(`${baseUrl}/collections/${collectionName}/points`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ points: batch }),
        });

        if (!upsertResponse.ok) {
            const msg = await upsertResponse.text();
            throw new Error(`${LOG_PREFIX} Upsert failed for batch starting at ${b}: ${msg}`);
        }

        console.log(`${LOG_PREFIX} Upserted batch ${Math.floor(b / BATCH_SIZE) + 1} (${batch.length} vectors).`);
    }

    console.log(`${LOG_PREFIX} Indexing complete for "${characterName}": ${points.length} vector(s) in "${collectionName}".`);
    return points.length;
}


// =============================================================================
// Exports
// =============================================================================

export {
    deriveCollectionName,
    embedText,
    ensureCollection,
    wipeCollection,
    wipeSessions,
    indexSummaries,
    DEFAULT_QDRANT_PORT,
    DEFAULT_EMBEDDING_PORT,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_SIZE,
};
