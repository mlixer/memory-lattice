// =============================================================================
// summarizer.js — Memory Pipeline: Core Summarization Module
// =============================================================================
// Responsibilities:
//   1. Load raw chat sessions for a character from the ST backend
//   2. Chunk sessions into digestible segments
//   3. Format chunks as clean transcripts for the LLM
//   4. Summarize chunks (LLM call)
//   5. Extract structured facts (LLM call)
//   6. Merge new facts into the existing fact sheet
// =============================================================================

'use strict';

// SillyTavern attaches a CSRF token to every /api/... request. Using
// getRequestHeaders() returns { Content-Type, X-CSRF-Token } so our calls
// authenticate properly — otherwise ST responds with HTTP 403.
import { getRequestHeaders } from '../../../../script.js';

// --- Constants ---------------------------------------------------------------

const DEFAULT_CHUNK_SIZE = 30; // messages per chunk, configurable in settings
const LOG_PREFIX = '[MemoryPipeline]';

// =============================================================================
// SECTION 0: Fact Categories & Prompt Templates (single source of truth)
// =============================================================================
//
// Categories and prompts are now DATA, not hardcoded structure. The settings
// UI (index.js) exposes both for editing; everything in this module receives
// the active categories/prompts as parameters and falls back to these
// defaults when none are provided — so existing callers keep working and
// behavior is unchanged until the user actually edits something.
//
// Category shape: { key, label, description }
//   key         — the JSON key the LLM emits and the fact sheet stores.
//                 Lowercase [a-z0-9_]. RENAMING a key requires migrating
//                 existing sheets (the settings UI handles this).
//   label       — human-readable heading used at context-injection time.
//   description — one-line guidance shown to the extraction LLM.
//
// Prompt templates may contain tokens that the code expands at call time:
//   {{CATEGORIES}}            → "key: description" list, one per line
//   {{CATEGORY_SCHEMA}}       → JSON skeleton { "key": [], ... }
//   {{CATEGORY_SCHEMA_FACTS}} → JSON skeleton with {fact, added} entries
//   {{CATEGORY_COUNT}}        → number of active categories
//   {{CAP}}                   → per-category fact cap (compact prompt)
//   {{TODAY}}                 → current date, YYYY-MM-DD (any prompt; useful
//                               for time-aware compaction experiments)
//
// Load-bearing tokens that a user edit removed are appended back at call
// time (with a console warning) so a broken edit degrades instead of
// breaking the parsers.

const DEFAULT_FACT_CATEGORIES = [
    { key: 'people',            label: 'people',            description: 'Relationships, people, and environmental factors.' },
    { key: 'world',            label: 'world',         description: 'home, places, possessions, recurring context.' },
    { key: 'preferences', label: 'preferences', description: 'likes, dislikes, habits, style.' },
    { key: 'projects',                label: 'projects',                description: 'ongoing work and its current state.' },
    { key: 'identity',              label: 'identity',              description: 'Who the user is: name, pronouns, essentials.' },
];

/** Returns the active category list, falling back to defaults. */
function resolveCategories(categories) {
    return (Array.isArray(categories) && categories.length > 0)
        ? categories
        : DEFAULT_FACT_CATEGORIES;
}

/** "key: description" list for {{CATEGORIES}}. */
function buildCategoriesBlock(categories) {
    return resolveCategories(categories)
        .map(c => `${c.key}: ${c.description ?? ''}`.trim())
        .join('\n');
}

/** Empty-array JSON skeleton for {{CATEGORY_SCHEMA}} (extraction). */
function buildCategorySchema(categories) {
    const lines = resolveCategories(categories).map(c => `    "${c.key}": []`);
    return `{\n${lines.join(',\n')}\n}`;
}

/** {fact, added} JSON skeleton for {{CATEGORY_SCHEMA_FACTS}} (compaction). */
function buildCategoryFactsSchema(categories) {
    const lines = resolveCategories(categories)
        .map(c => `  "${c.key}":[{ "fact": string, "added": string }, ...]`);
    return `{\n${lines.join(',\n')}\n}`;
}

/**
 * Expands all known tokens in a prompt template. Unknown text is left
 * untouched; templates without tokens pass through unchanged.
 */
function substitutePromptTokens(template, { cap, categories } = {}) {
    if (typeof template !== 'string') return template;
    const cats = resolveCategories(categories);
    const substitutions = {
        '{{TODAY}}':                 new Date().toISOString().split('T')[0],
        '{{CAP}}':                   cap !== undefined && cap !== null ? String(cap) : '{{CAP}}',
        '{{CATEGORIES}}':            buildCategoriesBlock(cats),
        '{{CATEGORY_SCHEMA}}':       buildCategorySchema(cats),
        '{{CATEGORY_SCHEMA_FACTS}}': buildCategoryFactsSchema(cats),
        '{{CATEGORY_COUNT}}':        String(cats.length),
    };
    let out = template;
    for (const [token, value] of Object.entries(substitutions)) {
        out = out.split(token).join(value);
    }
    return out;
}


// =============================================================================
// SECTION 1: Data Loading
// =============================================================================

/**
 * Resolves a Unix-ms timestamp for a chat entry from ST's API response.
 *
 * ST's chat list responses can include several time-bearing fields and the
 * shape isn't fully consistent — `last_mes` is the most common but isn't
 * always present (e.g. when `simple: true` is passed). When none of the
 * expected fields are usable, we fall back to parsing the date out of the
 * chat filename, which always follows the pattern:
 *
 *   "<character> - YYYY-MM-DD@HHhMMmSSsmmsms.jsonl"
 *
 * @param {object} chat       - One entry from /api/characters/chats
 * @returns {number}          - Unix ms timestamp, or NaN if nothing usable
 */
function getChatTimestamp(chat) {
    // Try the standard ST fields in order of preference.
    const candidates = [chat.last_mes, chat.chat_create_date, chat.create_date];
    for (const v of candidates) {
        if (!v) continue;
        // Numeric timestamps (ST sometimes returns ms-since-epoch directly).
        if (typeof v === 'number' && Number.isFinite(v) && v > 0) {
            // Heuristic: a value below ~10^12 is likely seconds, not ms.
            return v < 1e12 ? v * 1000 : v;
        }
        const parsed = new Date(v).getTime();
        if (Number.isFinite(parsed)) return parsed;
    }

    // Fallback: extract YYYY-MM-DD from the filename.
    if (typeof chat.file_name === 'string') {
        const match = chat.file_name.match(/(\d{4})-(\d{2})-(\d{2})@(\d{2})h(\d{2})m(\d{2})s/);
        if (match) {
            const [, yyyy, mm, dd, hh, mi, ss] = match;
            const isoLike = `${yyyy}-${mm}-${dd}T${hh}:${mi}:${ss}`;
            const parsed = new Date(isoLike).getTime();
            if (Number.isFinite(parsed)) return parsed;
        }
        // Even less precise fallback — just the date portion.
        const dateOnly = chat.file_name.match(/(\d{4}-\d{2}-\d{2})/);
        if (dateOnly) {
            const parsed = new Date(dateOnly[1]).getTime();
            if (Number.isFinite(parsed)) return parsed;
        }
    }

    return NaN;
}


/**
 * Fetches all chat sessions for a character that have been modified since a
 * given timestamp. Returns sessions sorted oldest-first so summarization runs
 * in chronological order.
 *
 * @param {string} characterName  - Display name of the character (e.g. "Aria")
 * @param {string} avatarUrl      - Character avatar filename (e.g. "Aria.png")
 *                                  ST uses this as the unique character key.
 * @param {number} since          - Unix timestamp in ms. Only sessions modified
 *                                  after this time are returned. Pass 0 to load
 *                                  all sessions (used on first run).
 * @returns {Promise<Session[]>}  - Array of session objects, each containing:
 *                                  { fileName, lastModified, messages[] }
 */
async function loadChatsForCharacter(characterName, avatarUrl, since = 0) {
    // -------------------------------------------------------------------------
    // Step 1: Get the list of chat files for this character from ST's backend.
    // ST returns metadata for each chat: file_name, last_mes, chat_items, etc.
    // -------------------------------------------------------------------------
    let chatList;

    try {
        const listResponse = await fetch('/api/characters/chats', {
            method: 'POST',
            headers: getRequestHeaders(),
            body: JSON.stringify({
                avatar_url: avatarUrl,
                // simple:true strips last_mes — we need timestamps to filter
                // by recency, so request the full response.
                simple: false,
            }),
        });

        if (!listResponse.ok) {
            throw new Error(`HTTP ${listResponse.status}`);
        }

        chatList = await listResponse.json();
    } catch (err) {
        throw new Error(`${LOG_PREFIX} Could not fetch chat list for "${characterName}": ${err.message}`);
    }

    if (!Array.isArray(chatList) || chatList.length === 0) {
        console.log(`${LOG_PREFIX} No chats found for "${characterName}".`);
        return [];
    }

    console.log(`${LOG_PREFIX} Chat list returned ${chatList.length} entries for "${characterName}".`);

    // -------------------------------------------------------------------------
    // Step 2: Filter to only sessions modified after the `since` timestamp.
    // Prefer ST's last_mes field when present (most accurate). When it's
    // missing or unparseable, fall back to extracting the date from the
    // chat filename, which follows the pattern:
    //   "<character> - YYYY-MM-DD@HHhMMmSSsmmsms.jsonl"
    // -------------------------------------------------------------------------
    const newChats = chatList.filter(chat => {
        const lastModifiedMs = getChatTimestamp(chat);
        if (!Number.isFinite(lastModifiedMs)) {
            console.warn(`${LOG_PREFIX} Could not determine timestamp for "${chat.file_name}", skipping.`);
            return false;
        }
        // Attach the resolved timestamp so we don't recompute it later.
        chat._resolvedMs = lastModifiedMs;
        return lastModifiedMs > since;
    });

    if (newChats.length === 0) {
        console.log(`${LOG_PREFIX} No new or modified chats for "${characterName}" since last index.`);
        return [];
    }

    console.log(`${LOG_PREFIX} Found ${newChats.length} session(s) to process for "${characterName}".`);

    // -------------------------------------------------------------------------
    // Step 3: Load the full message list for each qualifying session.
    // We iterate sequentially (not Promise.all) to avoid hammering the backend.
    // -------------------------------------------------------------------------
    const loadedSessions = [];

    for (const chat of newChats) {
        let messages;

        try {
            // ST's /api/chats/get expects the filename WITHOUT the .jsonl
            // extension — it appends the extension server-side. Sending the
            // full filename results in a silent 200 OK with an empty array.
            const fileNameWithoutExt = chat.file_name.replace(/\.jsonl$/, '');

            const chatResponse = await fetch('/api/chats/get', {
                method: 'POST',
                headers: getRequestHeaders(),
                credentials: 'include',
                body: JSON.stringify({
                    ch_name: characterName,
                    file_name: fileNameWithoutExt,
                    avatar_url: avatarUrl,
                }),
            });

            if (!chatResponse.ok) {
                throw new Error(`HTTP ${chatResponse.status}`);
            }

            messages = await chatResponse.json();
        } catch (err) {
            // A single failed chat load should not abort the whole pipeline.
            console.warn(`${LOG_PREFIX} Skipping session "${chat.file_name}": ${err.message}`);
            continue;
        }

        // Normalise response shape — ST sometimes returns the messages
        // wrapped in a parent object instead of a plain array.
        if (!Array.isArray(messages)) {
            const candidates = ['messages', 'chat', 'data', 'items'];
            for (const k of candidates) {
                if (Array.isArray(messages?.[k])) {
                    messages = messages[k];
                    break;
                }
            }
        }

        if (!Array.isArray(messages)) {
            console.warn(`${LOG_PREFIX} Could not coerce response into a message array for "${chat.file_name}", skipping.`);
            continue;
        }

        console.log(`${LOG_PREFIX} Loaded ${messages.length} raw messages from "${chat.file_name}".`);

        // ---------------------------------------------------------------------
        // Clean the message list:
        //   - Keep only turns that have a defined is_user flag (real dialogue)
        //   - Drop empty or whitespace-only messages
        //   - Drop ST system/instruction entries (no name, or [START] markers)
        // ---------------------------------------------------------------------
        const cleanMessages = messages.filter(m =>
            m.is_user !== undefined &&
            typeof m.mes === 'string' &&
            m.mes.trim().length > 0 &&
            !m.mes.trim().startsWith('[') // strip [START], [INST], etc.
        );

        // DEBUG: when cleaning drops everything, log per-condition reject counts
        // so we can see which check is responsible.
        if (cleanMessages.length === 0 && messages.length > 0) {
            let noIsUser = 0, badMes = 0, emptyMes = 0, bracketMes = 0;
            for (const m of messages) {
                if (m?.is_user === undefined) { noIsUser++; continue; }
                if (typeof m.mes !== 'string') { badMes++; continue; }
                if (m.mes.trim().length === 0) { emptyMes++; continue; }
                if (m.mes.trim().startsWith('[')) { bracketMes++; continue; }
            }
            console.log(`${LOG_PREFIX} [DEBUG] Cleaning rejects: noIsUser=${noIsUser}, badMes=${badMes}, emptyMes=${emptyMes}, bracketMes=${bracketMes}`);
        }

        if (cleanMessages.length === 0) {
            console.log(`${LOG_PREFIX} Session "${chat.file_name}" had no usable messages after cleaning, skipping.`);
            continue;
        }

        loadedSessions.push({
            fileName: chat.file_name,
            lastModified: chat._resolvedMs,
            messages: cleanMessages,
        });
    }

    // Sort oldest-first so the LLM sees events in chronological order.
    loadedSessions.sort((a, b) => a.lastModified - b.lastModified);

    console.log(`${LOG_PREFIX} Loaded ${loadedSessions.length} session(s) for "${characterName}".`);
    return loadedSessions;
}


// =============================================================================
// SECTION 2: Chunking
// =============================================================================

/**
 * Splits an array of messages into fixed-size chunks.
 * The last chunk may be smaller than chunkSize.
 *
 * Why chunk at all? A single long session could be 200+ messages. Sending
 * that to the LLM in one call risks hitting context limits and produces
 * worse summaries (models lose detail on long inputs). Smaller chunks keep
 * each LLM call focused and accurate.
 *
 * @param {object[]} messages   - Cleaned message array from loadChatsForCharacter
 * @param {number}   chunkSize  - Max messages per chunk (default: 30)
 * @returns {object[][]}        - Array of chunks (each chunk is an array of messages)
 */
function chunkSession(messages, chunkSize = DEFAULT_CHUNK_SIZE, gapMs = 0) {
    if (!Array.isArray(messages) || messages.length === 0) return [];

    const chunks = [];
    let current = [];
    let prevTs = null;

    for (const msg of messages) {
        const ts = parseMessageTimestamp(msg);

        // Start a new chunk when the size cap is reached, or when there's a
        // real-time gap between consecutive messages (gapMs=0 disables gap
        // splitting → pure fixed-size chunking, the v1 behavior). Gap
        // splitting keeps each chunk a temporally-coherent "moment": a chat
        // file that spans a lunch break, an evening, or several days no
        // longer gets its unrelated moments summarized as one blur.
        const gapExceeded = gapMs > 0
            && prevTs !== null
            && ts !== null
            && (ts - prevTs) > gapMs;

        if (current.length >= chunkSize || (gapExceeded && current.length > 0)) {
            chunks.push(current);
            current = [];
        }

        current.push(msg);
        if (ts !== null) prevTs = ts;
    }

    if (current.length > 0) chunks.push(current);
    return chunks;
}

/**
 * Best-effort parse of an ST message's send_date into Unix ms.
 *
 * ST has stored several formats over time:
 *   - number (ms since epoch, occasionally seconds)
 *   - ISO-ish strings
 *   - humanized strings like "June 19, 2026 2:20pm" (no space before am/pm,
 *     which some Date parsers reject — we insert one defensively)
 *
 * @param {object} msg  - ST message object
 * @returns {number|null} - Unix ms, or null when unparseable/absent
 */
function parseMessageTimestamp(msg) {
    const v = msg?.send_date;
    if (v === undefined || v === null) return null;

    if (typeof v === 'number' && Number.isFinite(v) && v > 0) {
        return v < 1e12 ? v * 1000 : v; // seconds vs ms heuristic
    }

    if (typeof v === 'string' && v.trim().length > 0) {
        let parsed = new Date(v).getTime();
        if (Number.isFinite(parsed)) return parsed;

        // Humanized format: insert a space before am/pm and retry.
        const spaced = v.replace(/(\d)(am|pm)\b/i, '$1 $2');
        parsed = new Date(spaced).getTime();
        if (Number.isFinite(parsed)) return parsed;
    }

    return null;
}

/**
 * Resolves a chunk's own timestamp: the send_date of its FIRST message
 * (when the moment began), falling back to the session-level timestamp
 * when messages carry no usable send_date.
 *
 * @param {object[]} chunk       - Array of ST message objects
 * @param {number}   fallbackMs  - Session lastModified ms
 * @returns {number}             - Unix ms
 */
function resolveChunkTimestamp(chunk, fallbackMs) {
    for (const msg of chunk) {
        const ts = parseMessageTimestamp(msg);
        if (ts !== null) return ts;
    }
    return fallbackMs;
}


// =============================================================================
// SECTION 3: Transcript Formatting
// =============================================================================

/**
 * Converts a chunk of ST message objects into a clean, readable transcript
 * string. This is what gets sent to the LLM — the model never sees raw ST
 * JSON, only this formatted text.
 *
 * Format:
 *   User: [message text]
 *   Aria: [message text]
 *   User: [message text]
 *   ...
 *
 * Why preserve original wording? This is the key anti-contamination measure.
 * The LLM receives the raw dialogue verbatim and is instructed to extract/
 * compress, not reinterpret. The closer the transcript is to the source, the
 * less room there is for the summarization model's bias to creep in.
 *
 * @param {object[]} chunk          - Array of ST message objects
 * @param {string}   characterName  - Used to label the assistant turns
 * @returns {string}                - Formatted transcript ready for LLM input
 */
function formatChunkAsTranscript(chunk, characterName) {
    return chunk
        .map(msg => {
            const speaker = msg.is_user ? 'User' : characterName;
            // Trim each message but preserve internal line breaks so multi-
            // paragraph responses aren't collapsed into unreadable walls of text.
            return `${speaker}: ${msg.mes.trim()}`;
        })
        .join('\n\n');
}



// =============================================================================
// SECTION 4: LLM API Call — Shared Helper
// =============================================================================

/**
 * Sends a system + user prompt to the llama.cpp OpenAI-compatible endpoint.
 *
 * Why temperature 0.1?
 * Deterministic, consistent extraction — not creative variation. Low temperature
 * keeps the model at its most confident output, which for structured extraction
 * is almost always correct.
 *
 * @param {string} systemPrompt
 * @param {string} userPrompt
 * @param {number} port       - llama.cpp container port (configurable, default 8070)
 * @param {number} maxTokens  - completion budget. Summaries fit in ~2048;
 *                              long structured JSON output (extractFacts)
 *                              needs much more headroom — bump per call.
 * @param {boolean} jsonMode  - when true, sends response_format:json_object so
 *                              llama.cpp constrains sampling to valid JSON.
 *                              Prevents common errors like missing commas.
 * @returns {Promise<string>}
 */
async function callLlm(systemPrompt, userPrompt, port = 8070, maxTokens = 49152, jsonMode = false) {
    const url = `http://localhost:${port}/v1/chat/completions`;

    const body = {
        model: 'local',   // llama.cpp ignores this field
        messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user',   content: userPrompt   },
        ],
        temperature: 0.1,
        max_tokens: maxTokens,
        stream: false,
        // Qwen3.6 thinks by default and ignores the /no_think soft switch
        // that Qwen3 supported. The official escape is to pass
        // enable_thinking=false through chat_template_kwargs, which
        // llama.cpp forwards to the Jinja template. With this set,
        // reasoning_content stays empty and the model writes directly
        // into content — which is what our pipeline needs.
        chat_template_kwargs: { enable_thinking: false },
    };

    if (jsonMode) {
        // Constrains llama.cpp's sampler to valid JSON. Prevents missing-comma
        // and other common malformations that Qwen3.6 produces routinely.
        body.response_format = { type: 'json_object' };
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });

    if (!response.ok) {
        throw new Error(`${LOG_PREFIX} LLM API error: HTTP ${response.status}`);
    }

    const data = await response.json();
    const choice = data?.choices?.[0];
    const text = choice?.message?.content;

    if (!text) {
        // Dump everything useful before failing so we can diagnose:
        //   - finish_reason ('length' = hit max_tokens, 'stop' = clean stop)
        //   - whether the model returned only <think> tokens
        //   - usage stats showing where tokens went
        console.warn(`${LOG_PREFIX} LLM returned empty content. Full diagnostics:`);
        console.warn(`${LOG_PREFIX} - finish_reason:`, choice?.finish_reason);
        console.warn(`${LOG_PREFIX} - usage:`, data?.usage);
        console.warn(`${LOG_PREFIX} - reasoning_content (if present):`,
                     choice?.message?.reasoning_content?.slice?.(0, 500));
        console.warn(`${LOG_PREFIX} - full response:`, data);
        throw new Error(`${LOG_PREFIX} LLM returned empty response.`);
    }

    // Even when content is non-empty, finish_reason="length" means we hit
    // max_tokens and output is silently truncated. JSON callers will crash
    // on parse; prose callers get cut off mid-sentence. Warn so we notice
    // and can raise the budget for the next run.
    if (choice?.finish_reason === 'length') {
        console.warn(`${LOG_PREFIX} LLM hit max_tokens (${maxTokens}); output truncated. usage=`, data?.usage);
    }

    return text.trim();
}


// =============================================================================
// SECTION 5: Summarization
// =============================================================================

// --- Prompt Design Notes -----------------------------------------------------
//
//  1. System prompt enforces strict extraction-only mode. The model is told
//     explicitly what it must NOT do — no opinions, no inference, no invented
//     detail. This is the anti-contamination layer.
//
//  2. Session date is injected into the user prompt so the model reasons about
//     recency naturally ("this happened 6 months ago") without a scoring system.
//
//  3. Qwen3.6 thinking is disabled at the API layer in callLlm() via
//     chat_template_kwargs.enable_thinking=false. We want extraction, not
//     reasoning. Thinking burns the completion budget with no benefit here.
//
//  4. Output is plain prose — no headers, no bullet lists. Dense prose produces
//     better embeddings; vector similarity works on semantic flow, not structure.

const DEFAULT_SUMMARIZE_PROMPT = `You are recording a personal memory of a conversation you had with the user. Your goal is to preserve the truth of the exchange, prioritizing accuracy and the specific language used over narrative flourishes.
Guidelines for the Memory:

Grounded Reality: Base all descriptions and emotional states exclusively on what is explicitly stated or shown in the transcript. Do not imagine sensory details or internal states that are not provided.
Verbatim Fidelity: Whenever possible, use the user's exact words when documenting their thoughts, feelings, or instructions. The nuance of their language is the most important part of the memory.
Emotional Evidence: Instead of describing an emotion you assume they feel, quote the words they used to describe that emotion. (Example: Instead of "they were anxious," write "they said they felt 'like their heart was going to explode'.")
The Narrative Sequence: Document the interaction as it unfolded, from the first message to the last, ensuring the cause-and-effect of the conversation is clear.
First-Person Perspective: Write the memory from your own perspective, focusing on your responses to what the user shared.

Rules:

Do not mention that this is a summary.
Write the memory directly, in the past tense.
There is no length limit; write as much as is necessary to be accurate and complete.`;

/**
 * Summarizes a single chunk of conversation into a dense memory string.
 *
 * @param {string} transcript    - From formatChunkAsTranscript()
 * @param {string} sessionDate   - ISO date string e.g. "2026-04-29"
 * @param {string} characterName
 * @param {number} port
 * @param {string} [promptOverride] - Custom system prompt from settings;
 *                                    falls back to the shipped default.
 * @returns {Promise<string>}    - Summary string ready to embed into Qdrant
 */
async function summarizeChunk(transcript, sessionDate, characterName, port = 8070, promptOverride = null) {
    const systemPrompt = (typeof promptOverride === 'string' && promptOverride.trim().length > 0)
        ? promptOverride
        : DEFAULT_SUMMARIZE_PROMPT;

    const userPrompt = `This conversation with ${characterName} took place on ${sessionDate}.
Summarize the key events, topics, and anything significant. Be concise and factual.

--- TRANSCRIPT BEGIN ---
${transcript}
--- TRANSCRIPT END ---`;

    const summary = await callLlm(systemPrompt, userPrompt, port);
    console.log(`${LOG_PREFIX} Summarized chunk (${transcript.length} chars → ${summary.length} chars)`);
    return summary;
}


// =============================================================================
// SECTION 5.5: Session-Level Refinement (two-pass summarization)
// =============================================================================
//
// Per-chunk summaries are produced in isolation — chunk 4 has no idea what
// chunk 1 covered. A topic that spans multiple chunks gets re-explained
// every time; a name revealed late never propagates back to earlier
// summaries; redundant setup gets repeated. This is the "summaries produced
// in isolation" problem (Known Issue B from v1).
//
// `refineSessionSummaries` is the fix: after a session's per-chunk
// summaries are all generated, we feed them as one ordered list to the LLM
// and ask it to rewrite each one with awareness of the full session arc.
// The refined versions are what get indexed into Qdrant; the raw versions
// are throwaway intermediates.
//
// Design constraints:
//   - SAME COUNT in and out. Refining is not merging.
//   - SAME ORDER and SAME chunkIndex values. Caller relies on these.
//   - NO NEW INFORMATION. Refiner can only use what's in the input
//     summaries, not invent details. Fact extraction has already pulled
//     the durable signal from the raw transcripts before refinement runs.
//   - Best-effort. Any failure (parse error, wrong count, malformed
//     entries) falls back to the raw summaries for that session.
//
// Cost: one extra LLM call per session, regardless of chunk count. For
// most session sizes the input + output fit comfortably in 8192 tokens.
// Very long sessions (>~25 chunks) are skipped — refinement quality
// degrades when the model can't hold the whole arc in attention, and the
// output budget can be exhausted. Above the threshold, raw summaries are
// indexed as-is, which is the v1 behaviour.
//
// Fact extraction is UNCHANGED. Facts come from raw transcript chunks,
// not from refined summaries — refining is lossy by design and the fact
// sheet needs maximum signal density. Refinement only affects Qdrant.
// =============================================================================

// Above this many chunks per session we skip refinement. Inputs + outputs
// for a session of N chunks scale roughly linearly with N, and beyond ~25
// the model loses cross-chunk attention quality and the output cap gets
// risky. Configurable per-call.
const MAX_CHUNKS_FOR_REFINEMENT = 25;

// The output-format block is kept as its own constant so that, when a user
// edit accidentally deletes the {"refined": [...]} instruction, the call
// site can append it back (the parser depends on that shape).
const DEFAULT_REFINE_OUTPUT_BLOCK = `OUTPUT FORMAT:
Return ONLY raw JSON with this exact structure (no preamble, no markdown):
{
  "refined": [
    { "chunkIndex": <int>, "text": "<refined summary>" },
    ...
  ]
}
The "refined" array length must equal the input array length. The chunkIndex values must match the input one-to-one in order.`;

const DEFAULT_REFINE_PROMPT = `You are synthesizing a series of memory fragments from a single conversation with the user into a cohesive record. Your goal is to ensure that the final version is both accurate and self-contained, serving as a reliable record for future reference.
Your Job:

Read all fragments to ensure the chronology is correct and that later revelations are factored into the earlier portions for full clarity.
Rewrite each fragment as a standalone scene. Each memory must be encapsulated—meaning it should contain enough context to be understood on its own without needing the fragments before or after it.
Remove redundant information that is repeated across fragments, but do not remove details that add specific substance to the memory.
Ensure that all dialogue and emotional descriptions are derived directly from the source fragments.

Rules:

Return exactly one refined summary for each input fragment.
Maintain the original chunkIndex values and JSON structure.
Write in the first person, past tense.
Do not invent details, scenarios, or emotions not present in the source material.
Output ONLY raw JSON.

${DEFAULT_REFINE_OUTPUT_BLOCK}`;

/**
 * Refines a session's per-chunk summaries by running them all through the
 * LLM together, so each refined summary has cross-chunk context. Best-
 * effort — falls back to the input summaries on any failure.
 *
 * @param {object[]} rawSummaries  - Array of { chunkIndex, text }, in chunk order
 * @param {string}   sessionDate   - ISO date string e.g. "2026-04-29"
 * @param {string}   characterName
 * @param {number}   port
 * @param {string}   [promptOverride] - Custom system prompt from settings.
 *                                      If it lacks the "refined" output-shape
 *                                      instruction, the default output block
 *                                      is appended so the parser still works.
 * @returns {Promise<object[]>}    - Refined array (same shape and count), or
 *                                   the input array unchanged on failure
 */
async function refineSessionSummaries(rawSummaries, sessionDate, characterName, port = 8070, promptOverride = null) {
    if (!Array.isArray(rawSummaries) || rawSummaries.length === 0) {
        return rawSummaries;
    }

    // Single-chunk sessions have nothing to cross-reference.
    if (rawSummaries.length < 2) {
        return rawSummaries;
    }

    // Very long sessions blow past the model's reliable cross-chunk
    // attention AND risk exhausting the output budget. Fall back gracefully.
    if (rawSummaries.length > MAX_CHUNKS_FOR_REFINEMENT) {
        console.log(`${LOG_PREFIX} Refinement skipped: ${rawSummaries.length} chunks exceeds threshold ${MAX_CHUNKS_FOR_REFINEMENT}. Indexing raw summaries.`);
        return rawSummaries;
    }

    // Resolve the system prompt: custom if provided, default otherwise.
    // If a custom prompt no longer mentions the "refined" wrapper key the
    // parser needs, append the shipped output block (warn — degraded edit).
    let systemPrompt = (typeof promptOverride === 'string' && promptOverride.trim().length > 0)
        ? promptOverride
        : DEFAULT_REFINE_PROMPT;
    if (!systemPrompt.includes('"refined"')) {
        console.warn(`${LOG_PREFIX} Custom refine prompt is missing the {"refined": [...]} output instruction — appending the default output block so parsing still works.`);
        systemPrompt = `${systemPrompt}\n\n${DEFAULT_REFINE_OUTPUT_BLOCK}`;
    }

    const userPrompt = `This conversation with ${characterName} took place on ${sessionDate}.

The following ${rawSummaries.length} summaries are sequential chunks of that single conversation, in order. Refine them as instructed in your system message.

INPUT:
${JSON.stringify(rawSummaries, null, 2)}`;

    let raw;
    try {
        raw = await callLlm(systemPrompt, userPrompt, port, 49152, true);
    } catch (err) {
        console.warn(`${LOG_PREFIX} Refinement LLM call failed: ${err.message}. Indexing raw summaries.`);
        return rawSummaries;
    }

    let cleaned = raw.replace(/```json|```/g, '').trim();
    let parsed;
    try {
        parsed = JSON.parse(cleaned);
    } catch (err) {
        // Same Qwen missing-comma repair pattern used elsewhere.
        const repaired = cleaned.replace(/"\s*\n\s*"/g, '",\n"');
        try {
            parsed = JSON.parse(repaired);
            console.log(`${LOG_PREFIX} Refinement JSON repair successful (inserted missing commas).`);
        } catch (err2) {
            console.warn(`${LOG_PREFIX} Refinement produced unparseable JSON. Indexing raw summaries. Error: ${err.message}`);
            return rawSummaries;
        }
    }

    // The prompt asks for an object of shape { "refined": [...] }. This
    // matches the json_object semantics that response_format enforces in
    // OpenAI-compatible mode — top-level must be an object — so model and
    // constraint pull in the same direction. We still accept a few legacy
    // alias keys and a bare top-level array, both as defensive fallbacks
    // in case the model gets creative with the wrapper key.
    let arr;
    if (Array.isArray(parsed))                       arr = parsed;
    else if (Array.isArray(parsed?.refined))         arr = parsed.refined;
    else if (Array.isArray(parsed?.summaries))       arr = parsed.summaries;
    else if (Array.isArray(parsed?.chunks))          arr = parsed.chunks;
    else if (Array.isArray(parsed?.items))           arr = parsed.items;

    if (!Array.isArray(arr)) {
        // Diagnostic: surface the top-level shape so if a model picks yet
        // another wrapper key, the fix is to add it to the alias list
        // rather than having to dig through the raw response.
        const shape = (parsed && typeof parsed === 'object' && !Array.isArray(parsed))
            ? `object with keys: ${Object.keys(parsed).join(', ') || '(none)'}`
            : `type ${typeof parsed}`;
        console.warn(`${LOG_PREFIX} Refinement output was not a recognised array shape (got ${shape}). Indexing raw summaries.`);
        return rawSummaries;
    }

    // Structural validation: same count, same chunkIndex sequence in
    // order, non-empty text on every entry. Anything off → fall back. We
    // are strict because mis-indexed summaries would corrupt the Qdrant
    // metadata permanently.
    if (arr.length !== rawSummaries.length) {
        console.warn(`${LOG_PREFIX} Refinement returned ${arr.length} summaries; expected ${rawSummaries.length}. Indexing raw.`);
        return rawSummaries;
    }
    for (let i = 0; i < arr.length; i++) {
        const entry = arr[i];
        if (typeof entry?.text !== 'string' || entry.text.trim().length === 0) {
            console.warn(`${LOG_PREFIX} Refinement output has empty/non-string text at index ${i}. Indexing raw.`);
            return rawSummaries;
        }
        // chunkIndex must match the input in order. Be lenient about type
        // (some models emit it as a string).
        const refinedIdx = typeof entry.chunkIndex === 'number'
            ? entry.chunkIndex
            : parseInt(entry.chunkIndex, 10);
        if (refinedIdx !== rawSummaries[i].chunkIndex) {
            console.warn(`${LOG_PREFIX} Refinement chunkIndex mismatch at position ${i} (got ${refinedIdx}, expected ${rawSummaries[i].chunkIndex}). Indexing raw.`);
            return rawSummaries;
        }
    }

    // Build the refined output array using validated entries. Normalise
    // text trimming so downstream embedding gets clean input.
    const refined = arr.map((entry, i) => ({
        chunkIndex: rawSummaries[i].chunkIndex,
        text: entry.text.trim(),
    }));

    // Length-change accounting for the log line — useful to spot when the
    // refiner is being aggressive about de-duplication versus expansive.
    const totalRawChars     = rawSummaries.reduce((s, x) => s + x.text.length, 0);
    const totalRefinedChars = refined.reduce((s, x) => s + x.text.length, 0);
    const delta = totalRefinedChars - totalRawChars;
    const sign  = delta >= 0 ? '+' : '';
    console.log(`${LOG_PREFIX} Refined ${refined.length} summaries (${totalRawChars} → ${totalRefinedChars} chars, ${sign}${delta}).`);

    return refined;
}


// =============================================================================
// SECTION 6: Fact Extraction
// =============================================================================

// --- Prompt Design Notes -----------------------------------------------------
//
//  Fact extraction is a separate LLM call from summarization — deliberately.
//  Mixing both tasks in one prompt produces worse results for both: the model
//  splits attention and facts become less precise, summaries less clean.
//
//  The model returns raw JSON with no preamble. Backticks are stripped
//  defensively in the parser. Missing category keys are filled with [].
//
//  Each fact is stored as { fact, added } — a short statement plus a date.
//  The date enables the merge logic to preserve conflicting facts in order:
//  e.g. "lives in New York (2025-11-01)" + "lives in Chicago (2026-04-29)"

const DEFAULT_EXTRACT_FACTS_PROMPT = `You are the keeper of the durable facts about the user and your bond with them. Your task is to extract the most durable and significant facts from the conversation, ensuring they are recorded with enough context to be meaningful in the future.
Categories for Extraction:

{{CATEGORIES}}

Rules:

Contextual Completeness: A fact should not be a mere data point; it should be a statement. Instead of "likes cats," write "finds comfort in cats because they provide a non-judgmental presence."
Durability: Only extract facts that are likely to remain true over time.
No Inferences: Do not assume a fact is true because it seems implied; it must be explicitly stated by the user or established through the interaction.
Return ONLY raw JSON with the {{CATEGORY_COUNT}} categories.

Return this exact structure:
{{CATEGORY_SCHEMA}}`;

/**
 * Extracts structured facts from a transcript chunk.
 *
 * @param {string} transcript
 * @param {string} sessionDate
 * @param {string} characterName
 * @param {number} port
 * @param {string}   [promptOverride] - Custom system prompt TEMPLATE from
 *                                      settings (tokens are expanded here).
 * @param {object[]} [categories]     - Active category list from settings.
 * @returns {Promise<object>} - One array per active category key
 */
async function extractFacts(transcript, sessionDate, characterName, port = 8070, promptOverride = null, categories = null) {
    const cats = resolveCategories(categories);

    // Resolve the template, then expand tokens against the ACTIVE category
    // list — this is what keeps prompts and category edits in sync.
    const template = (typeof promptOverride === 'string' && promptOverride.trim().length > 0)
        ? promptOverride
        : DEFAULT_EXTRACT_FACTS_PROMPT;

    // The JSON schema is load-bearing (the parser fills the sheet from these
    // keys). If a custom prompt dropped the schema token AND doesn't seem to
    // spell out a structure of its own, append the schema so extraction
    // keeps returning keys we can read.
    let effectiveTemplate = template;
    if (!template.includes('{{CATEGORY_SCHEMA}}')) {
        const mentionsAllKeys = cats.every(c => template.includes(`"${c.key}"`) || template.includes(c.key));
        if (!mentionsAllKeys) {
            console.warn(`${LOG_PREFIX} Custom extract prompt is missing {{CATEGORY_SCHEMA}} and doesn't mention all category keys — appending the schema block so parsing still works.`);
            effectiveTemplate = `${template}\n\nReturn ONLY raw JSON with this exact structure:\n{{CATEGORY_SCHEMA}}`;
        }
    }

    const systemPrompt = substitutePromptTokens(effectiveTemplate, { categories: cats });

    const userPrompt = `Extract all facts from the following conversation with ${characterName} (date: ${sessionDate}).
Return only the JSON structure described in your instructions.

--- TRANSCRIPT BEGIN ---
${transcript}
--- TRANSCRIPT END ---`;

    // Fact extraction emits much longer JSON than summaries do — five
    // categories, each potentially many entries from a 100k-char chunk.
    // 8192 tokens leaves real headroom; 2048 truncates routinely.
    // jsonMode=true asks llama.cpp to constrain sampling to valid JSON.
    const raw = await callLlm(systemPrompt, userPrompt, port, 24576, true);

    // Strip markdown fences defensively in case the model adds them anyway.
    let cleaned = raw.replace(/```json|```/g, '').trim();

    let parsed;
    try {
        parsed = JSON.parse(cleaned);
    } catch (err) {
        // Attempt to repair the most common Qwen JSON malformation: missing
        // commas between adjacent string elements inside arrays. Pattern:
        //     "foo"
        //     "bar"           ← no comma between
        // Becomes:
        //     "foo",
        //     "bar"
        // The match requires only whitespace + newline between the two
        // strings, so it never affects valid "key": "value" pairs (those
        // have a colon between them) or properly comma-separated lists.
        const repaired = cleaned.replace(/"\s*\n\s*"/g, '",\n"');

        if (repaired !== cleaned) {
            try {
                parsed = JSON.parse(repaired);
                console.log(`${LOG_PREFIX} JSON repair successful (inserted missing commas).`);
            } catch (err2) {
                console.warn(`${LOG_PREFIX} Fact extraction returned unparseable JSON, skipping. Parse error: ${err.message}`);
                console.warn(`${LOG_PREFIX} Repair attempt also failed: ${err2.message}`);
                console.warn(`${LOG_PREFIX} Raw output length: ${raw.length} chars. Full raw:\n${raw}`);
                return emptyFactCategories(cats);
            }
        } else {
            console.warn(`${LOG_PREFIX} Fact extraction returned unparseable JSON, skipping. Parse error: ${err.message}`);
            console.warn(`${LOG_PREFIX} Raw output length: ${raw.length} chars. Full raw:\n${raw}`);
            return emptyFactCategories(cats);
        }
    }

    // Validate all expected keys; fill any missing ones with [].
    const result = emptyFactCategories(cats);
    for (const key of Object.keys(result)) {
        if (Array.isArray(parsed[key])) {
            result[key] = parsed[key].filter(f => typeof f === 'string' && f.trim().length > 0);
        }
    }

    const total = Object.values(result).reduce((sum, arr) => sum + arr.length, 0);
    console.log(`${LOG_PREFIX} Extracted ${total} fact(s) from chunk (date: ${sessionDate})`);
    return result;
}

/** Returns a clean empty fact category object for the active categories. */
function emptyFactCategories(categories = null) {
    const result = {};
    for (const c of resolveCategories(categories)) {
        result[c.key] = [];
    }
    return result;
}


// =============================================================================
// SECTION 7: Fact Sheet Merge
// =============================================================================

/**
 * Merges newly extracted facts into the existing fact sheet.
 *
 * Merge rules:
 *  - Each fact stored as { fact: string, added: string (ISO date) }
 *  - Exact duplicate facts (case-insensitive) are skipped
 *  - New non-duplicate facts are appended with their date
 *  - Existing facts are NEVER deleted — full history is preserved so
 *    conflicting facts appear side by side with dates, giving the model
 *    the context to understand change over time
 *
 * @param {object|null} existingSheet  - Current fact sheet, or null on first run
 * @param {object}      newFacts       - Output from extractFacts()
 * @param {string}      sessionDate    - ISO date to stamp onto new facts
 * @param {object[]}    [categories]   - Active category list from settings
 * @returns {object}                   - Updated fact sheet ready to save
 */
function mergeFactSheet(existingSheet, newFacts, sessionDate, categories = null) {
    const sheet = existingSheet ?? {
        character: null,
        last_updated: null,
        facts: emptyFactCategories(categories),
    };

    // Ensure all active categories exist even on older or partial sheets.
    // Keys already on the sheet that are NOT in the active list are left
    // untouched — never delete data here.
    for (const key of Object.keys(emptyFactCategories(categories))) {
        if (!Array.isArray(sheet.facts[key])) {
            sheet.facts[key] = [];
        }
    }

    let addedCount = 0;

    for (const category of Object.keys(newFacts)) {
        // Defensive: extraction should only emit active-category keys (all
        // ensured above), but never crash on an unexpected one.
        if (!Array.isArray(sheet.facts[category])) {
            sheet.facts[category] = [];
        }
        for (const newFact of newFacts[category]) {
            const normalised = newFact.trim().toLowerCase();

            const isDuplicate = sheet.facts[category].some(
                existing => existing.fact.trim().toLowerCase() === normalised
            );

            if (!isDuplicate) {
                sheet.facts[category].push({
                    fact: newFact.trim(),
                    added: sessionDate,
                });
                addedCount++;
            }
        }
    }

    sheet.last_updated = new Date().toISOString().split('T')[0];
    console.log(`${LOG_PREFIX} Fact sheet merge complete: ${addedCount} new fact(s) added.`);
    return sheet;
}

// =============================================================================
// SECTION 7.5: Fact Sheet Compaction (consolidate + cap)
// =============================================================================
//
// The merge step in Section 7 only catches case-insensitive exact-string
// duplicates. After multiple pipeline runs, the sheet accumulates facts that
// say the same thing in different words, where one is a strict subset of
// another, or where many entries about the same trait could be expressed as
// one richer statement. Left unchecked, the sheet grows past the LLM's
// dedup-pass budget and the system can't self-maintain.
//
// `compactFactSheet` runs ONE LLM call at the end of each pipeline run and
// does three things at once, per category, independently:
//
//   1. CONSOLIDATE — actively merge semantically-overlapping facts into one
//      richer fact (preserving the earliest "added" date).
//   2. DEDUPLICATE — for trivial duplicates / strict subsets, keep one.
//   3. CAP — return at most N facts per category; when over, keep the most
//      specific and durable ones.
//
// After the LLM call, deterministic post-processing enforces the cap as a
// safety net: if any category came back over the cap, we hard-truncate by
// keeping the entries with the EARLIEST `added` dates. Facts that have
// survived multiple compaction runs are reinforced over time and represent
// the most stable signal; new untested facts that arrived this run can be
// re-derived from the source chats on the next run if they were real.
//
// Best-effort: any failure (LLM error, parse error, structural mismatch)
// falls back to the input sheet unchanged — the pipeline never blocks here.
// =============================================================================

// Tokenized template — {{CAP}}, {{CATEGORY_SCHEMA_FACTS}} and
// {{CATEGORY_COUNT}} are expanded at call time against the ACTIVE
// category list, so category edits flow into this prompt automatically.
const DEFAULT_COMPACT_PROMPT = `You are the custodian of your shared history with the user. Your task is to ensure that the most essential truths of their identity, their journey, and your bond are preserved while removing redundancies.

    YOUR TASK — three operations, applied to each category:

        1.  WEAVE: When multiple facts describe a similar thread of the user's life or personality, weave them together into a single, richer narrative fact. Do not just merge them; synthesize the underlying truth. Use the earliest "added" date to preserve the origin of that realization.
        2. CLARIFY: For facts that are identical or near-identical, keep the one that possesses the most clarity and depth. Keep the earliest "added" date.
        3. DISTILL: You must keep the total facts in each category at or below {{CAP}}. To do this, you must distinguish between surface-level information and the core of who the user is. Prioritize the durable truths and the evolving dynamics of your relationship over the mundane.

    RULES YOU MUST FOLLOW:

   - Do NOT consolidate across different categories.
   - Do NOT invent details; every refined fact must be rooted in the provided input.
   - Do NOT remove facts that offer a unique perspective or a critical piece of the user's story.
   - Maintain the "YYYY-MM-DD" date format, using the earliest date from the merged group.
   - Output must be valid JSON.

OUTPUT FORMAT:
Return ONLY raw JSON with this exact structure (no preamble, no markdown):
{{CATEGORY_SCHEMA_FACTS}}
Always return all {{CATEGORY_COUNT}} categories, even if an array is empty.`;

/**
 * Runs an LLM compaction pass over a fact sheet: consolidates overlapping
 * facts, removes duplicates, and trims each category to at most `cap`
 * entries. Best-effort — falls back to the input sheet on any failure.
 *
 * Token budget is bounded by construction now: 5 categories x cap x ~20
 * tokens per fact entry. With cap=12 that's ~1200 tokens of output, well
 * within 8192. v1's dedup hit max_tokens routinely on established sheets;
 * the cap makes that unreachable.
 *
 * @param {object} sheet  - Fact sheet object (with .facts inside)
 * @param {number} cap    - Max facts per category (deterministic ceiling)
 * @param {number} port   - llama.cpp port
 * @returns {Promise<object>} - Same sheet shape, with compacted facts
 */
async function compactFactSheet(sheet, cap = 12, port = 8070, promptOverride = null, categories = null) {
    if (!sheet?.facts) return sheet;

    const cats = resolveCategories(categories);
    const catKeys = cats.map(c => c.key);

    const totalBefore = catKeys.reduce(
        (sum, cat) => sum + (sheet.facts[cat]?.length ?? 0), 0);

    // Skip the LLM call when there's nothing meaningful to compact.
    // A single-fact sheet can't be consolidated and can't exceed any
    // sensible cap.
    if (totalBefore < 2) {
        return sheet;
    }

    // Resolve the prompt template. If a custom prompt dropped the load-
    // bearing tokens, append them back (warn) so a broken edit degrades
    // instead of breaking the run.
    let template = (typeof promptOverride === 'string' && promptOverride.trim().length > 0)
        ? promptOverride
        : DEFAULT_COMPACT_PROMPT;
    if (!template.includes('{{CAP}}')) {
        console.warn(`${LOG_PREFIX} Custom compact prompt is missing {{CAP}} — appending the cap instruction.`);
        template += `\n\nYou must keep the total facts in each category at or below {{CAP}}.`;
    }
    if (!template.includes('{{CATEGORY_SCHEMA_FACTS}}')) {
        console.warn(`${LOG_PREFIX} Custom compact prompt is missing {{CATEGORY_SCHEMA_FACTS}} — appending the output schema so parsing still works.`);
        template += `\n\nOUTPUT FORMAT:\nReturn ONLY raw JSON with this exact structure (no preamble, no markdown):\n{{CATEGORY_SCHEMA_FACTS}}\nAlways return all {{CATEGORY_COUNT}} categories, even if an array is empty.`;
    }
    const systemPrompt = substitutePromptTokens(template, { cap, categories: cats });

    // Send ONLY the active categories to the LLM. Keys on the sheet that
    // are no longer in the active list (e.g. after a category was removed
    // or renamed without migration) are held back and re-attached to the
    // result untouched — the compactor must never be able to delete them.
    const activeFacts = {};
    for (const key of catKeys) {
        activeFacts[key] = Array.isArray(sheet.facts[key]) ? sheet.facts[key] : [];
    }

    const userPrompt = `Compact this fact sheet (max ${cap} per category):\n${JSON.stringify(activeFacts, null, 2)}`;

    let raw;
    try {
        raw = await callLlm(systemPrompt, userPrompt, port, 49152, true);
    } catch (err) {
        console.warn(`${LOG_PREFIX} Compact LLM call failed: ${err.message}. Keeping original sheet.`);
        return sheet;
    }

    let cleaned = raw.replace(/```json|```/g, '').trim();
    let parsed;
    try {
        parsed = JSON.parse(cleaned);
    } catch (err) {
        // Same Qwen missing-comma repair as extractFacts.
        const repaired = cleaned.replace(/"\s*\n\s*"/g, '",\n"');
        try {
            parsed = JSON.parse(repaired);
            console.log(`${LOG_PREFIX} Compact JSON repair successful (inserted missing commas).`);
        } catch (err2) {
            console.warn(`${LOG_PREFIX} Compact produced unparseable JSON. Keeping original sheet. Error: ${err.message}`);
            return sheet;
        }
    }

    // Structural validation — every category must be present and an array
    // of {fact, added} objects. If anything's off, fall back to the
    // original. We are strict here because corrupting the schema would
    // affect every future run.
    for (const cat of catKeys) {
        const arr = parsed[cat];
        if (!Array.isArray(arr)) {
            console.warn(`${LOG_PREFIX} Compact output missing category "${cat}". Keeping original sheet.`);
            return sheet;
        }
        for (const entry of arr) {
            if (typeof entry?.fact !== 'string' || typeof entry?.added !== 'string') {
                console.warn(`${LOG_PREFIX} Compact output has malformed entry in "${cat}". Keeping original sheet.`);
                return sheet;
            }
        }
    }

    // Deterministic cap enforcement — safety net for when the LLM ignores
    // its instruction (rare with json_mode + Qwen3.6, but it does happen).
    // Strategy: when a category is over-cap, keep entries with the EARLIEST
    // "added" dates first. Rationale: a fact that has survived through
    // multiple compaction runs is reinforced — the same observation has
    // been re-derived from source chats N times in a row — so it carries
    // more signal than a brand-new entry from this run. New untested
    // facts that get trimmed will reappear next run if they were genuine.
    const final = {};
    let trimmedAny = false;
    for (const cat of catKeys) {
        const arr = parsed[cat];
        if (arr.length <= cap) {
            final[cat] = arr;
        } else {
            // Sort ascending by added (lexicographic on YYYY-MM-DD works),
            // then keep the first `cap` entries.
            const sorted = [...arr].sort(
                (a, b) => (a.added || '').localeCompare(b.added || '')
            );
            final[cat] = sorted.slice(0, cap);
            trimmedAny = true;
            console.warn(`${LOG_PREFIX} Compact: category "${cat}" returned ${arr.length} > cap ${cap}; hard-truncated to ${cap} oldest-first.`);
        }
    }

    // Re-attach inactive keys verbatim so nothing is ever silently lost
    // to a category rename/removal that hasn't been migrated.
    for (const [key, value] of Object.entries(sheet.facts)) {
        if (!(key in final) && Array.isArray(value) && value.length > 0) {
            final[key] = value;
            console.log(`${LOG_PREFIX} Compact: preserved ${value.length} fact(s) under inactive category "${key}" (not sent to the LLM).`);
        }
    }

    const totalAfter = catKeys.reduce(
        (sum, cat) => sum + final[cat].length, 0);

    const note = trimmedAny ? ' (cap safety net engaged)' : '';
    console.log(`${LOG_PREFIX} Compact: ${totalBefore} → ${totalAfter} facts (${totalBefore - totalAfter} removed/merged)${note}.`);

    return {
        ...sheet,
        facts: final,
    };
}

// =============================================================================
// SECTION 8: Pipeline Orchestrator (fully wired)
// =============================================================================

/**
 * Entry point for the summarization pipeline for a single character.
 * Orchestrates: load → chunk → summarize (+ extract facts per chunk) →
 * refine summaries with cross-chunk context per session → return.
 *
 * Refinement (v2, Section 5.5) only affects what gets returned in
 * `summaries` (and thus indexed into Qdrant downstream). Fact extraction
 * still runs against the raw transcript chunks — refining is lossy by
 * design and we want maximum signal density in the fact sheet.
 *
 * @param {string}  characterName
 * @param {string}  avatarUrl
 * @param {number}  since           - Unix ms timestamp from last_indexed store
 * @param {number}  chunkSize       - From extension settings
 * @param {number}  port            - llama.cpp port
 * @param {object|null} existingFactSheet
 * @param {boolean} refineSummaries - When true (default), run the two-pass
 *                                    refinement at the end of each session.
 *                                    Single-chunk sessions and very long
 *                                    sessions skip refinement internally.
 * @returns {Promise<PipelineResult>}
 */
async function runPipeline(
    characterName,
    avatarUrl,
    since = 0,
    chunkSize = DEFAULT_CHUNK_SIZE,
    port = 8070,
    existingFactSheet = null,
    refineSummaries = true,
    promptOverrides = null,   // { summarize, extract, refine } — strings or null
    categories = null,        // active category list from settings
    chunkGapMs = 0,           // M4: split chunks on real-time gaps (0 = off)
) {
    console.log(`${LOG_PREFIX} Starting pipeline for "${characterName}"...`);

    const sessions = await loadChatsForCharacter(characterName, avatarUrl, since);

    if (sessions.length === 0) {
        console.log(`${LOG_PREFIX} Nothing to process. Pipeline complete.`);
        return { summaries: [], factSheet: existingFactSheet };
    }

    const summaries = [];
    let factSheet = existingFactSheet;

    for (const session of sessions) {
        const chunks = chunkSession(session.messages, chunkSize, chunkGapMs);
        const sessionDate = new Date(session.lastModified).toISOString().split('T')[0];
        console.log(`${LOG_PREFIX} Session "${session.fileName}": ${session.messages.length} messages → ${chunks.length} chunk(s)${chunkGapMs > 0 ? ' (gap-aware)' : ''}.`);

        // Per-session collector for raw summaries. We hold them in this
        // local array (not pushed straight to the outer `summaries`) so we
        // can run the refinement pass over the complete session at the end
        // of the inner loop. Fact extraction is unaffected — it runs per-
        // chunk against the raw transcript and merges as we go.
        const rawSessionSummaries = [];

        // M4: per-chunk timestamps. Each chunk resolves its own timestamp
        // from its first message's send_date (falling back to the session
        // timestamp). The chunk's own DATE is what the LLM is told the
        // conversation happened on, what facts get stamped with, and what
        // gets indexed — so a chat file spanning multiple days no longer
        // mislabels its earlier moments with the file's last-modified date.
        const chunkTimestamps = chunks.map(c => resolveChunkTimestamp(c, session.lastModified));

        for (let i = 0; i < chunks.length; i++) {
            const transcript = formatChunkAsTranscript(chunks[i], characterName);
            const chunkDate  = new Date(chunkTimestamps[i]).toISOString().split('T')[0];

            // Run summarization and fact extraction sequentially.
            // Local LLM backends (llama.cpp, etc.) typically serve one request
            // at a time. Running these in parallel can cause one call to come
            // back with empty content while the server is busy with the other.
            // Sequential is ~2x slower per chunk but reliable.
            const summary  = await summarizeChunk(transcript, chunkDate, characterName, port, promptOverrides?.summarize);
            const newFacts = await extractFacts(transcript, chunkDate, characterName, port, promptOverrides?.extract, categories);

            rawSessionSummaries.push({ chunkIndex: i, text: summary });
            factSheet = mergeFactSheet(factSheet, newFacts, chunkDate, categories);
        }

        // v2: optional two-pass refinement over all the session's summaries
        // at once. The refiner returns the same count and order, or falls
        // back to rawSessionSummaries unchanged on any failure — so this
        // path is always safe.
        const sessionSummaries = (refineSummaries && rawSessionSummaries.length > 1)
            ? await refineSessionSummaries(rawSessionSummaries, sessionDate, characterName, port, promptOverrides?.refine)
            : rawSessionSummaries;

        for (const s of sessionSummaries) {
            summaries.push({
                sessionFile: session.fileName,
                chunkIndex:  s.chunkIndex,
                sessionDate: new Date(chunkTimestamps[s.chunkIndex] ?? session.lastModified).toISOString().split('T')[0],
                timestampMs: chunkTimestamps[s.chunkIndex] ?? session.lastModified,
                text:        s.text,
            });
        }
    }

    // Stamp the character name onto the fact sheet on its first run.
    if (factSheet && !factSheet.character) {
        factSheet.character = characterName;
    }

    console.log(`${LOG_PREFIX} Pipeline complete. ${summaries.length} summary chunk(s) ready for indexing.`);
    return { summaries, factSheet };
}


// =============================================================================
// Exports
// =============================================================================

export {
    loadChatsForCharacter,
    chunkSession,
    formatChunkAsTranscript,
    callLlm,
    summarizeChunk,
    refineSessionSummaries,
    extractFacts,
    mergeFactSheet,
    compactFactSheet,
    runPipeline,
    parseMessageTimestamp,
    resolveChunkTimestamp,
    emptyFactCategories,
    substitutePromptTokens,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FACT_CATEGORIES,
    DEFAULT_SUMMARIZE_PROMPT,
    DEFAULT_EXTRACT_FACTS_PROMPT,
    DEFAULT_COMPACT_PROMPT,
    DEFAULT_REFINE_PROMPT,
};
