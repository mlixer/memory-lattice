// =============================================================================
// shelfIngest.js — Shelf ingest (Phase 4)
// =============================================================================
// The hands' own writing joins chat memory: curiosity entries (kind: 'hands') and
// outbox gifts (kind: 'gift', any lifecycle state) are indexed into the same
// per-character collection the pipeline already manages — the braid stays ONE
// entity: any chat can semantically retrieve a thought the companion had offline.
//
//   - point id: UUIDv5(SHELF_NS, 'shelf:'+relPath) — content-independent, so a
//     file keeps its point across edits and re-embeds
//   - the diff is by content hash: existing points carry sha256 in payload; we
//     embed + upsert only entries whose hash differs. No cursor file to drift.
//   - files that vanish from disk are deleted from the collection by point id.
//   - a fullWipe rebuild simply re-embeds everything from source.
// =============================================================================

import { embedText, ensureCollection, deriveCollectionName } from './qdrantIndexer.js';

const LOG_PREFIX = '[MemoryPipeline/Shelf]';

// URL namespace per RFC 4122 — distinct from the chunk namespace by construction.
const SHELF_NS = '6ba7b812-9dad-11d0-80c4-00c04fd430c8';

// Deterministic UUIDv5 via WebCrypto (ST extensions live in a browser).
async function shelfPointId(relPath, ns = SHELF_NS) {
    const hb = (h) => Uint8Array.from(h.match(/../g).map((b) => parseInt(b, 16)));
    const msg = new Uint8Array([
        ...hb(ns.replace(/-/g, '')),
        ...new TextEncoder().encode('shelf:' + relPath),
    ]);
    const h = new Uint8Array(await crypto.subtle.digest('SHA-1', msg)).subarray(0, 16);
    h[6] = (h[6] & 0x0f) | 0x50;   // version 5
    h[8] = (h[8] & 0x3f) | 0x80;   // RFC 4122 variant
    const x = [...h].map((b) => b.toString(16).padStart(2, '0')).join('');
    return `${x.slice(0, 8)}-${x.slice(8, 12)}-${x.slice(12, 16)}-${x.slice(16, 20)}-${x.slice(20)}`;
}

// Paged scroll of the collection's existing shelf points ->
// Map relPath -> { id, sha256 }.
async function scrollShelfPoints(collection, qdrantPort) {
    const out = new Map();
    let offset = null;
    do {
        const body = {
            filter: { must: [{ key: 'source', match: { value: 'shelf' } }] },
            limit: 256,
            with_payload: ['relPath', 'sha256'],
            with_vector: false,
        };
        if (offset !== null) body.offset = offset;
        const r = await fetch(
            `http://localhost:${qdrantPort}/collections/${encodeURIComponent(collection)}/points/scroll`,
            { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        if (!r.ok) throw new Error(`${LOG_PREFIX} shelf scroll failed: HTTP ${r.status}`);
        const d = await r.json();
        for (const p of d.result?.points || []) out.set(p.payload.relPath, { id: p.id, sha256: p.payload.sha256 });
        offset = d.result?.next_page_offset ?? null;
    } while (offset);
    return out;
}

/**
 * Fetch all shelf entries from the ticket server's /shelf, diff against the
 * existing shelf points in this character's collection by content hash, and
 * embed + upsert only what changed. Idempotent; safe to run every pipeline
 * pass. Payload mirrors the chat-memory fields the reader reads (type, kind,
 * source, text, timestamp) so shelf hits render beside chat hits.
 *
 * @returns {Promise<{indexed:number, deleted:number}>}
 */
export async function indexShelfForCharacter(characterName, opts) {
    const { serverUrl, qdrantPort, embeddingPort, vectorSize,
            embeddingModel, embeddingUrl, embeddingApi } = opts;
    const collection = deriveCollectionName(characterName);

    const res = await fetch(`${serverUrl}/shelf`);
    if (!res.ok) throw new Error(`${LOG_PREFIX} shelf fetch failed: HTTP ${res.status}`);
    const { entries } = await res.json();

    const existing = await scrollShelfPoints(collection, qdrantPort);
    const byRel = new Map(entries.map((e) => [e.relPath, e]));

    const toIndex = entries.filter((e) => existing.get(e.relPath)?.sha256 !== e.sha256);
    const goneIds = [...existing].filter(([rel]) => !byRel.has(rel)).map(([, v]) => v.id);

    const points = [];
    for (const e of toIndex) {
        let vector;
        try {
            vector = await embedText(e.content, embeddingPort, embeddingModel, embeddingUrl, embeddingApi);
        } catch (err) {
            console.warn(`${LOG_PREFIX} Skipping ${e.relPath} — embedding failed: ${err.message}`);
            continue;
        }
        points.push({
            id: await shelfPointId(e.relPath),
            vector,
            payload: {
                type: 'long_term',
                kind: e.kind,        // 'hands' (curiosity) | 'gift'
                source: 'shelf',
                relPath: e.relPath,
                text: e.content,
                sha256: e.sha256,
                timestamp: e.mtimeMs, // date tag at read time
            },
        });
    }

    if (points.length || goneIds.length) await ensureCollection(collection, vectorSize, qdrantPort);

    for (let b = 0; b < points.length; b += 100) {
        const r = await fetch(
            `http://localhost:${qdrantPort}/collections/${encodeURIComponent(collection)}/points`,
            { method: 'PUT', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ points: points.slice(b, b + 100) }) });
        if (!r.ok) throw new Error(`${LOG_PREFIX} shelf upsert failed: HTTP ${r.status}`);
    }
    if (goneIds.length) {
        const r = await fetch(
            `http://localhost:${qdrantPort}/collections/${encodeURIComponent(collection)}/points/delete`,
            { method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ points: goneIds }) });
        if (!r.ok) throw new Error(`${LOG_PREFIX} shelf delete failed: HTTP ${r.status}`);
    }
    return { indexed: points.length, deleted: goneIds.length };
}
