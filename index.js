// =============================================================================
// index.js — Memory Lattice: SillyTavern Extension Entry Point
// =============================================================================
// Wires together the WRITE PATH (summarizer.js, factSheet.js,
// qdrantIndexer.js, consolidator.js — the nightly pipeline, plus
// shelfIngest.js, Phase 4) and the READ PATH (reader.js — per-message
// recall) into one complete memory system.
//
// Responsibilities:
//   1. Register extension with ST and initialise settings
//   2. Render the settings UI panel
//   3. Expose a manual "Run Now" trigger
//   4. Run the daily scheduler
//   5. Orchestrate the full pipeline: load → summarise → index → save
//   6. Track which characters have been indexed (for scheduled runs)
//   7. Per-message injection: recalled memories + the fact sheet
//
// NOTE: the internal settings key remains 'memoryPipeline' so existing
// installs keep their settings across the rename to Memory Lattice.
// =============================================================================

// ST extensions live at /scripts/extensions/third-party/[name]/
// Two separate ST modules at different depths:
//   • extensions.js  → /scripts/extensions.js  (3 levels up)
//     exports: extension_settings, getContext
//   • script.js      → /script.js              (4 levels up)
//     exports: saveSettingsDebounced
import { extension_settings, getContext } from '../../../extensions.js';
import { saveSettingsDebounced }          from '../../../../script.js';

import { runPipeline, compactFactSheet,
         emptyFactCategories,
         DEFAULT_FACT_CATEGORIES,
         DEFAULT_SUMMARIZE_PROMPT,
         DEFAULT_EXTRACT_FACTS_PROMPT,
         DEFAULT_COMPACT_PROMPT,
         DEFAULT_REFINE_PROMPT }                from './summarizer.js';
import { loadFactSheet, saveFactSheet, finaliseAndSave,
         formatFactSheetForContext,
         getLastIndexed,
         backupFactSheet, pruneOldBackups }     from './factSheet.js';
import { indexSummaries }                       from './qdrantIndexer.js';
import { indexShelfForCharacter }                from './shelfIngest.js';
import { DEFAULT_READER_SETTINGS,
         injectRecalledMemories,
         previewRecall,
         countMemories,
         testQdrantConnection,
         testEmbeddingConnection }              from './reader.js';
import { runConsolidation,
         DEFAULT_CONSOLIDATE_PROMPT }           from './consolidator.js';

// =============================================================================
// Constants
// =============================================================================

const EXTENSION_NAME = 'memoryPipeline';
// In-array marker for the interceptor's fact-sheet injection (double-call
// guard). Distinct from reader.js's recall marker and state-bridge's markers.
const SHEET_MARKER = '__lattice_sheet';
const LOG_PREFIX     = '[MemoryPipeline]';

// =============================================================================
// Default Settings
// =============================================================================

const DEFAULT_SETTINGS = {
    enabled:             true,
    refineSummaries:     true,  // v2: two-pass cross-chunk refinement per session
    llmPort:             8070,   // llama.cpp summarization model
    embeddingPort:       11434,  // Ollama default (nomic-embed-text)
    qdrantPort:          6333,   // Qdrant vector database
    chunkSize:           30,     // messages per summary chunk
    chunkGapHours:       6,      // M4: start a new chunk after a real-time gap
                                 // of this many hours (0 = disable gap splits)
    vectorSize:          768,    // must match embedding model dimensions
    scheduledTime:       '03:00',// HH:MM — daily pipeline run time
    factsPerCategoryCap: 12,     // v2: hard ceiling per fact-sheet category
    backupsRetention:    0,      // v2: max pre-merge backups kept; 0 = unlimited
    trackedChars:        [],     // avatarUrls of characters the pipeline has run for
    shelfEnabled:         false, // Phase 4: curiosities/ + outbox/ become memory (kind hands/gift).
                                 // Requires the companion ticket server's /shelf endpoint —
                                 // leave off unless you run one (github.com/mlixer/ticket-server).
    shelfServer:          '',    // blank = auto: loopback locally, :8443 proxy on tailnet

    // v2.1: user-editable prompts. Defaults are the shipped prompt
    // templates from summarizer.js VERBATIM, so behavior is unchanged
    // until a prompt is actually edited. Tokens like {{CATEGORIES}} /
    // {{CATEGORY_SCHEMA}} / {{CAP}} are expanded at run time.
    summarizePrompt:     DEFAULT_SUMMARIZE_PROMPT,
    extractFactsPrompt:  DEFAULT_EXTRACT_FACTS_PROMPT,
    compactPrompt:       DEFAULT_COMPACT_PROMPT,
    refinePrompt:        DEFAULT_REFINE_PROMPT,

    // v2.2 (M2): cross-session consolidation — a derived second layer of
    // Qdrant points (kind: 'consolidated'), rebuilt from scratch on every
    // consolidation pass. Originals are never touched.
    consolidationEnabled:   true,
    consolidationThreshold: 0.80,  // cosine similarity to join a cluster
    consolidationMinCluster: 3,    // min memories per qualifying cluster
    consolidationMaxClusters: 12,  // max clusters consolidated per run
    consolidatePrompt:      DEFAULT_CONSOLIDATE_PROMPT,

    // v2.1: user-editable fact categories — the single source of truth
    // consumed by extraction, merge, compaction, and context injection.
    // { key, label, description } per category. Renaming a key via the
    // UI migrates existing fact sheets (backup taken first).
    factCategories:      DEFAULT_FACT_CATEGORIES,

    // v3.1: fact-sheet placement + big-run compaction cadence.
    sheetPosition:         0,   // 0 = pinned at the very top (classic); N>0 = N messages from the end
    compactEveryNSessions: 0,   // mid-run fact compaction on big runs; 0 = end-of-run only

    // v3 (Memory Lattice): the READ PATH — per-message recall settings.
    // Defined in reader.js; spread here so getSettings() migration gives
    // existing installs the new keys without touching anything else.
    ...DEFAULT_READER_SETTINGS,
};

// =============================================================================
// Settings Initialisation
// =============================================================================

function getSettings() {
    if (!extension_settings[EXTENSION_NAME]) {
        extension_settings[EXTENSION_NAME] = { ...DEFAULT_SETTINGS };
    }
    // Merge in any new defaults added in updates without overwriting user values.
    for (const [key, val] of Object.entries(DEFAULT_SETTINGS)) {
        if (extension_settings[EXTENSION_NAME][key] === undefined) {
            // Deep-clone array/object defaults (factCategories) so user
            // edits never mutate the shared DEFAULT_SETTINGS reference.
            extension_settings[EXTENSION_NAME][key] =
                (typeof val === 'object' && val !== null)
                    ? JSON.parse(JSON.stringify(val))
                    : val;
        }
    }
    return extension_settings[EXTENSION_NAME];
}

function saveSettings() {
    saveSettingsDebounced();
}

// =============================================================================
// Scheduler
// =============================================================================

let schedulerTimer = null;

/**
 * Calculates ms until the next occurrence of a HH:MM time string,
 * then sets a timeout to run the pipeline and reschedule.
 *
 * @param {string} timeString - "HH:MM" e.g. "03:00"
 */
function scheduleNextRun(timeString) {
    if (schedulerTimer) {
        clearTimeout(schedulerTimer);
        schedulerTimer = null;
    }

    const settings = getSettings();
    if (!settings.enabled) return;

    const [hours, minutes] = timeString.split(':').map(Number);

    const now  = new Date();
    const next = new Date();
    next.setHours(hours, minutes, 0, 0);

    // If the target time has already passed today, schedule for tomorrow.
    if (next <= now) {
        next.setDate(next.getDate() + 1);
    }

    const msUntilNext = next - now;
    const hUntil = (msUntilNext / 1000 / 60 / 60).toFixed(1);
    console.log(`${LOG_PREFIX} Next scheduled run in ${hUntil}h (at ${next.toLocaleTimeString()}).`);

    schedulerTimer = setTimeout(async () => {
        console.log(`${LOG_PREFIX} Scheduled run triggered.`);
        await runPipelineForAllTracked();
        // Reschedule for the same time tomorrow.
        scheduleNextRun(timeString);
    }, msUntilNext);
}

// =============================================================================
// Pipeline Orchestration
// =============================================================================

/**
 * Runs the full pipeline for a single character.
 * Load → Summarise → Index → Save fact sheet.
 *
 * @param {string} characterName
 * @param {string} avatarUrl
 * @returns {Promise<boolean>} - true on success, false on error
 */
// Phase 4: shelf files -> memory. Explicit setting wins; else mirror state-bridge's
// location-aware resolution (loopback locally, :8443 proxy on the tailnet).
function resolveShelfServer() {
    const explicit = [getSettings().shelfServer, extension_settings.state_bridge?.server]
        .find((s) => typeof s === 'string' && s.trim());
    if (explicit) return explicit.trim().replace(/\/+$/, '');
    const h = location.hostname;
    if (h === 'localhost' || h === '127.0.0.1') return 'http://127.0.0.1:8002';
    return `https://${h}:8443`;
}

async function runShelfStep(characterName, settings) {
    try {
        setStatus(`Shelf-ingest for ${characterName}...`);
        const s = await indexShelfForCharacter(characterName, {
            serverUrl:      resolveShelfServer(),
            qdrantPort:     settings.qdrantPort,
            embeddingPort:  settings.embeddingPort,
            embeddingModel: settings.embeddingModel,
            embeddingUrl:   settings.embeddingUrl,
            embeddingApi:   settings.embeddingApi,
            vectorSize:     settings.vectorSize,
        });
        console.log(`${LOG_PREFIX} Shelf ingest for "${characterName}": ${s.indexed} indexed, ${s.deleted} removed.`);
    } catch (err) {
        console.warn(`${LOG_PREFIX} Shelf ingest failed (non-blocking):`, err);
    }
}

async function runPipelineForCharacter(characterName, avatarUrl, forceFullReindex = false) {
    const settings = getSettings();

    setStatus(`Running for ${characterName}...`);
    console.log(`${LOG_PREFIX} Starting full pipeline for "${characterName}" (${avatarUrl})`);

    try {
        // 1. Load the existing fact sheet (null on first run).
        const existingSheet = await loadFactSheet(avatarUrl);

        // forceFullReindex overrides the incremental filter: since=0 makes
        // the pipeline reprocess EVERY session and (via indexSummaries'
        // fullWipe branch) rebuild the whole Qdrant collection. The fact
        // sheet is NOT reset — it's loaded, re-merged, and compacted as
        // usual, so forcing a rebuild is safe for the sheet.
        const since = forceFullReindex ? 0 : getLastIndexed(existingSheet);

        if (forceFullReindex) {
            console.log(`${LOG_PREFIX} FORCED full reindex — reprocessing ALL chats for "${characterName}" (fact sheet preserved, Qdrant rebuilt).`);
        } else if (since > 0) {
            console.log(`${LOG_PREFIX} Incremental run — processing chats since ${new Date(since).toLocaleDateString()}`);
        } else {
            console.log(`${LOG_PREFIX} First run — processing all chats for "${characterName}"`);
        }

        // 1a. Take a deep-clone snapshot of the loaded sheet BEFORE the
        //     pipeline mutates it. mergeFactSheet (called inside runPipeline)
        //     pushes new entries directly onto existingSheet.facts[cat], so
        //     by the time we'd otherwise want to back up, the "previous"
        //     state is already gone. The clone preserves it cleanly.
        //
        //     We don't actually upload the snapshot yet — we wait until we
        //     know the pipeline did real work, to avoid creating useless
        //     backups on no-op runs.
        const preMergeSnapshot = existingSheet
            ? JSON.parse(JSON.stringify(existingSheet))
            : null;

        // 2. Run summarisation pipeline: load chats → chunk → summarise → extract facts.
        const { summaries, factSheet } = await runPipeline(
            characterName,
            avatarUrl,
            since,
            settings.chunkSize,
            settings.llmPort,
            existingSheet,
            settings.refineSummaries,
            {
                summarize: settings.summarizePrompt,
                extract:   settings.extractFactsPrompt,
                refine:    settings.refinePrompt,
                compact:   settings.compactPrompt,
            },
            settings.factCategories,
            Math.max(0, (settings.chunkGapHours ?? 0)) * 60 * 60 * 1000,
            settings.compactEveryNSessions ?? 0,
            settings.factsPerCategoryCap,
        );

        if (summaries.length === 0 && since > 0) {
            setStatus(`No new chats for ${characterName}. Done.`);
            console.log(`${LOG_PREFIX} No new summaries — skipping chat index step.`);
            // No chat mutation → no backup needed → snapshot is discarded. The shelf
            // may still have changed, though — it indexes independently of chats.
            if (settings.shelfEnabled) await runShelfStep(characterName, settings);
            return true;
        }

        // 3. Back up the pre-merge snapshot before going further. We're now
        //    committed to writing a new sheet, so we want the prior state
        //    preserved on disk. Skipped automatically when there was no
        //    prior sheet (first run for this character). NEVER throws —
        //    a failed backup logs a warning but does not block the pipeline.
        if (preMergeSnapshot) {
            await backupFactSheet(avatarUrl, preMergeSnapshot);
            // Enforce retention if configured. Currently a no-op until ST's
            // user-file delete endpoint is verified (see factSheet.js
            // SECTION 7). Default backupsRetention=0 means unlimited, in
            // which case this returns immediately without doing anything.
            await pruneOldBackups(avatarUrl, settings.backupsRetention);
        }

        // 4. Index summaries into Qdrant.
        //    Wipe strategy depends on the run type:
        //      - First run / forced rebuild (since === 0): the summaries cover
        //        every session, so a FULL collection wipe is correct and also
        //        clears any orphaned points from deleted chats.
        //      - Incremental run (since > 0): only some sessions were
        //        reprocessed. A full wipe would destroy dormant sessions'
        //        memories that won't be re-indexed, so we pass fullWipe=false
        //        and only the reprocessed sessions get wiped + replaced.
        const indexedCount = await indexSummaries(
            characterName,
            summaries,
            settings.qdrantPort,
            settings.embeddingPort,
            settings.vectorSize,
            settings.embeddingModel,
            since === 0,      // fullWipe only on first run / forced rebuild
            settings.embeddingUrl,
            settings.embeddingApi,
        );

        // 4a. (Phase 4) Shelf ingest: the hands' own writing (kind hands/gift) joins this
        //     same collection. Runs AFTER step 4 so a fullWipe rebuild has already
        //     cleared + re-inserted chat points. Best-effort — never blocks.
        if (settings.shelfEnabled) await runShelfStep(characterName, settings);

        // 4b. (M2) Consolidation pass: rebuild the derived layer of cross-
        //     session topic memories now that the base points are current.
        //     Best-effort — a failure here NEVER blocks the pipeline; the
        //     previous consolidated layer simply remains in place.
        if (settings.consolidationEnabled && indexedCount > 0) {
            try {
                setStatus(`Consolidating cross-session memories for ${characterName}...`);
                const cRes = await runConsolidation(characterName, {
                    qdrantPort:     settings.qdrantPort,
                    embeddingPort:  settings.embeddingPort,
                    embeddingModel: settings.embeddingModel,
                    embeddingUrl:   settings.embeddingUrl,
                    embeddingApi:   settings.embeddingApi,
                    llmPort:        settings.llmPort,
                    threshold:      settings.consolidationThreshold,
                    minCluster:     settings.consolidationMinCluster,
                    maxClusters:    settings.consolidationMaxClusters,
                    promptOverride: settings.consolidatePrompt,
                });
                console.log(`${LOG_PREFIX} Consolidation: ${cRes.indexed} topic memor${cRes.indexed === 1 ? 'y' : 'ies'} from ${cRes.basePoints} base points.`);
            } catch (err) {
                console.warn(`${LOG_PREFIX} Consolidation failed (non-blocking):`, err);
            }
        }

        // 5. Run the compaction pass on the merged fact sheet (v2 replacement
        //    for the v1 dedup-only pass). Three things happen in one LLM
        //    call: semantically-overlapping facts are merged into richer
        //    statements, trivial duplicates are removed, and each category
        //    is trimmed to the configured per-category cap. The cap also
        //    bounds the LLM's output budget, which was the v1 failure mode
        //    on established sheets. Best-effort — falls back to the input
        //    sheet on any error so the pipeline never blocks here.
        const sheetToSave = await compactFactSheet(
            factSheet ?? {
                character: characterName,
                facts: emptyFactCategories(settings.factCategories),
            },
            settings.factsPerCategoryCap,
            settings.llmPort,
            settings.compactPrompt,
            settings.factCategories,
        );

        // 6. Stamp timestamp and save the updated fact sheet.
        const savedSheet = await finaliseAndSave(avatarUrl, sheetToSave);

        // 7. Track this character for future scheduled runs.
        trackCharacter(avatarUrl);

        setStatus(`Done — ${indexedCount} memories indexed for ${characterName}.`);
        console.log(`${LOG_PREFIX} Pipeline complete for "${characterName}": ${indexedCount} vector(s) indexed.`);
        return true;

    } catch (err) {
        console.error(`${LOG_PREFIX} Pipeline failed for "${characterName}":`, err);
        setStatus(`Error for ${characterName}: ${err.message}`);
        return false;
    }
}

/**
 * Runs the pipeline for every character that has been indexed before.
 * Used by the daily scheduler.
 */
async function runPipelineForAllTracked() {
    const settings = getSettings();
    const tracked  = settings.trackedChars ?? [];

    if (tracked.length === 0) {
        console.log(`${LOG_PREFIX} No tracked characters — nothing to run on schedule.`);
        return;
    }

    console.log(`${LOG_PREFIX} Scheduled run for ${tracked.length} character(s).`);

    // We need characterName for each avatarUrl.
    // Pull it from ST's character list.
    const context    = getContext();
    const characters = context.characters ?? [];

    for (const avatarUrl of tracked) {
        const char = characters.find(c => c.avatar === avatarUrl);
        const name = char?.name ?? avatarUrl.replace(/\.[^/.]+$/, '');
        await runPipelineForCharacter(name, avatarUrl);
    }
}

/**
 * Adds an avatarUrl to the tracked characters list if not already present.
 */
function trackCharacter(avatarUrl) {
    const settings = getSettings();
    if (!settings.trackedChars.includes(avatarUrl)) {
        settings.trackedChars.push(avatarUrl);
        saveSettings();
        console.log(`${LOG_PREFIX} Now tracking "${avatarUrl}" for scheduled runs.`);
    }
}

// =============================================================================
// Status Display
// =============================================================================

/**
 * Updates the status line in the extension's UI panel.
 */
function setStatus(message) {
    const el = document.getElementById('memPipeline_status');
    if (el) el.textContent = message;
}

// =============================================================================
// Settings UI
// =============================================================================

// ST extensions use a collapsible "inline-drawer" structure so the settings
// panel integrates with the Extensions tab's expand/collapse behaviour.
// applyInlineDrawerListeners() (called in init) wires up the toggle chevron.
const SETTINGS_HTML = `
<div class="inline-drawer" id="memPipeline_settings">
    <div class="inline-drawer-toggle inline-drawer-header">
        <b>Memory Lattice</b>
        <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
    </div>
    <div class="inline-drawer-content">

        <div style="display:flex; align-items:center; gap:20px; margin-bottom:12px; flex-wrap:wrap;">
            <label style="display:flex; align-items:center; gap:6px;">
                <input type="checkbox" id="memPipeline_enabled" />
                Enabled
            </label>
            <label style="display:flex; align-items:center; gap:6px;"
                   title="Two-pass summarization: after per-chunk summaries, refine them all together so each one has cross-chunk context. Costs one extra LLM call per session.">
                <input type="checkbox" id="memPipeline_refineSummaries" />
                Refine summaries (cross-chunk context)
            </label>
            <label style="display:flex; align-items:center; gap:6px;"
                   title="M2: cluster related memories across sessions by embedding similarity and merge each cluster into one consolidated topic memory (a derived layer of Qdrant points, rebuilt every run — originals untouched). One LLM call per cluster.">
                <input type="checkbox" id="memPipeline_consolidationEnabled" />
                Consolidate cross-session topics
            </label>
        </div>

        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px 16px; margin-bottom:12px;">

            <label>Summarisation Model Port
                <input type="number" id="memPipeline_llmPort"
                       class="text_pole" style="width:100%; margin-top:2px;" />
            </label>

            <label>Embedding Model Port
                <input type="number" id="memPipeline_embeddingPort"
                       class="text_pole" style="width:100%; margin-top:2px;" />
            </label>

            <label>Qdrant Port
                <input type="number" id="memPipeline_qdrantPort"
                       class="text_pole" style="width:100%; margin-top:2px;" />
            </label>

            <label>Vector Size
                <input type="number" id="memPipeline_vectorSize"
                       class="text_pole" style="width:100%; margin-top:2px;" />
            </label>

            <label title="Embedding model name, used by BOTH the nightly write path and per-message recall. Changing it requires a Force Full Reindex — mixed-model vectors don't compare.">Embedding Model
                <input type="text" id="memPipeline_embeddingModel"
                       class="text_pole" style="width:100%; margin-top:2px;" placeholder="nomic-embed-text" />
            </label>

            <label title="Wire format of the embedding server: 'ollama' (/api/embeddings) or 'openai' (/v1/embeddings — llama.cpp's server speaks this).">Embedding API (ollama | openai)
                <input type="text" id="memPipeline_embeddingApi"
                       class="text_pole" style="width:100%; margin-top:2px;" placeholder="ollama" />
            </label>

            <label title="Full Qdrant base URL. Blank = http://localhost:&lt;port&gt;. These requests run in YOUR BROWSER, so when ST is loaded from another device, blank points at that device — set a reachable URL for remote use.">Qdrant URL (blank = localhost)
                <input type="text" id="memPipeline_qdrantUrl"
                       class="text_pole" style="width:100%; margin-top:2px;" placeholder="http://127.0.0.1:6333" />
            </label>

            <label title="Full embedding-server base URL. Blank = http://localhost:&lt;port&gt;. Same browser-side caveat as the Qdrant URL.">Embedding URL (blank = localhost)
                <input type="text" id="memPipeline_embeddingUrl"
                       class="text_pole" style="width:100%; margin-top:2px;" placeholder="http://127.0.0.1:11434" />
            </label>

            <label title="Where the fact sheet lands: 0 = pinned at the very top (classic). N&gt;0 = N messages from the end, nearer the current turn.">Fact Sheet Position (0 = top)
                <input type="number" id="memPipeline_sheetPosition"
                       class="text_pole" style="width:100%; margin-top:2px;" min="0" max="100" />
            </label>

            <label title="On long runs (especially Force Full Reindex), compact the fact sheet every N sessions instead of only once at the end — keeps the sheet inside the compaction call's budget. 0 = end-of-run only.">Compact Facts every N Sessions (0 = end only)
                <input type="number" id="memPipeline_compactEveryNSessions"
                       class="text_pole" style="width:100%; margin-top:2px;" min="0" max="100" />
            </label>

            <label>Shelf Server (blank = auto: loopback / :8443)
                <input type="text" id="memPipeline_shelfServer"
                       class="text_pole" style="width:100%; margin-top:2px;" placeholder="auto" />
            </label>

            <label style="display:flex; align-items:center; gap:6px;">
                <input type="checkbox" id="memPipeline_shelfEnabled" />
                <span>Shelf ingest (curiosities + gifts → memory)</span>
            </label>

            <label>Chunk Size (messages)
                <input type="number" id="memPipeline_chunkSize"
                       class="text_pole" style="width:100%; margin-top:2px;"
                       min="10" max="200" />
            </label>

            <label title="M4: start a new chunk when consecutive messages are separated by more than this many hours of real time, so one chunk never blurs together separate moments from a long-running chat file. 0 disables gap splitting. Requires a Force Full Reindex to apply to existing history.">Chunk Gap Split (hours, 0 = off)
                <input type="number" id="memPipeline_chunkGapHours"
                       class="text_pole" style="width:100%; margin-top:2px;"
                       min="0" max="720" step="0.5" />
            </label>

            <label>Daily Run Time (HH:MM)
                <input type="text" id="memPipeline_scheduledTime"
                       class="text_pole" style="width:100%; margin-top:2px;"
                       placeholder="03:00" pattern="[0-2][0-9]:[0-5][0-9]" />
            </label>

            <label title="Hard cap per fact-sheet category after compaction. Increase if you find 12 too restrictive; the compact pass bounds itself to this number per category.">Facts per Category Cap
                <input type="number" id="memPipeline_factsPerCategoryCap"
                       class="text_pole" style="width:100%; margin-top:2px;"
                       min="1" max="100" />
            </label>

            <label title="Maximum number of pre-merge fact-sheet backups to retain per character. 0 means unlimited (recommended for now — pruning is not yet wired up to ST's file API).">Backups Retention (0 = unlimited)
                <input type="number" id="memPipeline_backupsRetention"
                       class="text_pole" style="width:100%; margin-top:2px;"
                       min="0" max="9999" />
            </label>

            <label title="Cosine similarity required for a memory to join a topic cluster. Higher = tighter, fewer clusters; lower = looser, more clusters. Tune by pressing Rebuild Consolidated and reading the console.">Consolidation Similarity (0.5–0.99)
                <input type="number" id="memPipeline_consolidationThreshold"
                       class="text_pole" style="width:100%; margin-top:2px;"
                       min="0.5" max="0.99" step="0.01" />
            </label>

            <label title="Minimum memories a cluster needs (spanning at least 2 sessions) to be consolidated.">Consolidation Min Cluster Size
                <input type="number" id="memPipeline_consolidationMinCluster"
                       class="text_pole" style="width:100%; margin-top:2px;"
                       min="2" max="50" />
            </label>

            <label title="Maximum clusters consolidated per run (largest first). Bounds LLM calls.">Consolidation Max Clusters/Run
                <input type="number" id="memPipeline_consolidationMaxClusters"
                       class="text_pole" style="width:100%; margin-top:2px;"
                       min="1" max="100" />
            </label>

        </div>

        <div style="display:flex; gap:8px; margin-bottom:10px;">
            <button id="memPipeline_saveSettings" class="menu_button">
                Save Settings
            </button>
            <button id="memPipeline_runNow" class="menu_button menu_button_default">
                &#9654; Run Now (current character)
            </button>
        </div>

        <div style="display:flex; gap:8px; margin-bottom:10px; flex-wrap:wrap;">
            <button id="memPipeline_forceReindex" class="menu_button"
                    title="Reprocess ALL chat sessions from scratch and fully rebuild this character's Qdrant collection. Use to recover memories or after big changes. The fact sheet is preserved (re-merged and compacted). Slow — makes many LLM calls.">
                &#10227; Force Full Reindex (current character)
            </button>
            <button id="memPipeline_rebuildConsolidated" class="menu_button"
                    title="Run ONLY the consolidation pass now: cluster existing memories and rebuild the consolidated layer. No summarization, no fact extraction. Ideal for iterating on the consolidation prompt and similarity threshold.">
                &#8635; Rebuild Consolidated (current character)
            </button>
        </div>


        <div style="display:flex; gap:8px; margin-bottom:10px; flex-wrap:wrap;">
            <button id="memPipeline_editSheet" class="menu_button"
                    title="Open the current character's fact sheet for direct editing — quick corrections without waiting for the nightly merge. A backup is taken before every save.">
                &#9998; Edit Fact Sheet (current character)
            </button>
        </div>
        <div id="memPipeline_sheetEditorWrap" style="display:none; margin-bottom:10px;">
            <div style="font-size:0.8em; opacity:0.7; margin-bottom:4px;">
                Facts as JSON: { "category": [ { "fact": "...", "added": "YYYY-MM-DD" }, ... ] }.
                Edit, add, or delete entries. A backup of the current sheet is saved before your edit is written.
            </div>
            <textarea id="memPipeline_sheetEditor" class="text_pole" rows="16" spellcheck="false"
                      style="font-family:monospace; font-size:0.85em;"></textarea>
            <div style="display:flex; gap:8px; margin-top:6px;">
                <button id="memPipeline_sheetSave" class="menu_button menu_button_default">Save Fact Sheet</button>
                <button id="memPipeline_sheetCancel" class="menu_button">Cancel</button>
            </div>
        </div>

        <hr style="opacity:0.2; margin:14px 0 10px;">
        <div style="margin-bottom:10px;">
            <b>Recall (read path)</b>
            <div style="font-size:0.8em; opacity:0.7; margin:2px 0 8px;">
                Per-message memory retrieval. Tune with <b>Preview Recall</b> below — it shows
                exactly what would be injected, with similarity scores and token cost.
            </div>

            <div style="display:flex; align-items:center; gap:20px; margin-bottom:10px; flex-wrap:wrap;">
                <label style="display:flex; align-items:center; gap:6px;"
                       title="Master switch for per-message recall. Off = no retrieval, no injection (the fact sheet is controlled separately by the main Enabled toggle).">
                    <input type="checkbox" id="memPipeline_recallEnabled" />
                    Inject recalled memories
                </label>
                <span style="font-size:0.85em; opacity:0.75;">
                    Indexed memories: <b id="memPipeline_memoryCount">—</b>
                    <button id="memPipeline_refreshCount" class="menu_button" style="margin-left:6px;"
                            title="Count the points in the current character's Qdrant collection.">&#8635;</button>
                </span>
            </div>

            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px 16px; margin-bottom:10px;">
                <label title="How many memories are retrieved per message. Fewer = more focused recall and fewer tokens; which ones make the cut is governed by the relevance threshold.">Number of Memories (k)
                    <input type="number" id="memPipeline_recallK"
                           class="text_pole" style="width:100%; margin-top:2px;" min="1" max="50" />
                </label>
                <label title="Minimum cosine similarity for a memory to be injected at all. Raise it if irrelevant memories sneak in; lower it if obviously-relevant ones go missing. Use Preview Recall to see real scores from your corpus before changing this.">Relevance Threshold (0–1)
                    <input type="number" id="memPipeline_recallThreshold"
                           class="text_pole" style="width:100%; margin-top:2px;" min="0" max="1" step="0.05" />
                </label>
                <label title="Where the memory block lands: this many messages from the end of the chat. Smaller = closer to the new message (stronger influence).">Memory Position (depth)
                    <input type="number" id="memPipeline_recallPosition"
                           class="text_pole" style="width:100%; margin-top:2px;" min="0" max="100" />
                </label>
                <label title="How many messages BEFORE your new one are folded into the retrieval query. 0 = embed only the new message. Raise to 1–2 if short follow-ups ('and then?') retrieve poorly.">Retain Recent Messages
                    <input type="number" id="memPipeline_recallRetainRecent"
                           class="text_pole" style="width:100%; margin-top:2px;" min="0" max="20" />
                </label>
                <label title="Temporal neighbor expansion: each retrieved memory may bring its nearest-in-time neighbors from within ± this many days, so recall arrives with its context attached. 0 disables expansion.">Neighbor Time Window (days)
                    <input type="number" id="memPipeline_recallNeighborWindowDays"
                           class="text_pole" style="width:100%; margin-top:2px;" min="0" max="365" />
                </label>
                <label title="Upper bound on neighbors brought along per retrieved memory.">Max Neighbors per Match
                    <input type="number" id="memPipeline_recallNeighborsPerMatch"
                           class="text_pole" style="width:100%; margin-top:2px;" min="0" max="10" />
                </label>
                <label title="Upper bound on neighbors across the whole injection, regardless of per-match allowance.">Max Neighbors Total
                    <input type="number" id="memPipeline_recallNeighborsTotal"
                           class="text_pole" style="width:100%; margin-top:2px;" min="0" max="20" />
                </label>
            </div>

            <label title="The line that introduces recalled memories to the model. As with every prompt here: once your companion has a settled voice, consider asking it to rewrite this in its own words.">Recall header
                <textarea id="memPipeline_recallHeader" class="text_pole" rows="2" spellcheck="false"></textarea>
            </label>

            <div style="display:flex; gap:8px; margin:10px 0; flex-wrap:wrap;">
                <button id="memPipeline_testQdrant" class="menu_button"
                        title="Checks that Qdrant answers at the configured port.">Test Qdrant</button>
                <button id="memPipeline_testEmbedding" class="menu_button"
                        title="Sends a tiny embedding request and reports the vector size (should match Vector Size above).">Test Embedding</button>
            </div>

            <div style="display:flex; gap:8px; margin-bottom:6px;">
                <input type="text" id="memPipeline_previewQuery" class="text_pole" style="flex:1;"
                       placeholder="Type a test question, e.g. 'what did we do the day we finished the deck project?'" />
                <button id="memPipeline_previewRecall" class="menu_button menu_button_default"
                        title="Runs the full read path for this query — retrieval, threshold, neighbor expansion — and shows what WOULD be injected, with scores and token cost. Nothing is sent to the model.">Preview Recall</button>
                <button id="memPipeline_previewWide" class="menu_button"
                        title="Zoomed-out tuning view: top 20 memories by raw similarity, IGNORING your threshold and k, below-threshold items flagged. Use this to see where a memory you KNOW is relevant actually ranks.">Wide (top 20)</button>
            </div>
            <pre id="memPipeline_previewOutput"
                 style="display:none; white-space:pre-wrap; font-size:0.8em; max-height:300px; overflow-y:auto; border:1px solid rgba(128,128,128,0.3); border-radius:4px; padding:8px; margin:0 0 6px;"></pre>
        </div>

        <details id="memPipeline_advanced" style="margin:12px 0;">
            <summary><b>Advanced: Prompts &amp; Fact Categories</b></summary>

            <div style="margin-top:10px;">
                <b>Fact Categories</b>
                <div style="font-size:0.8em; opacity:0.7; margin:4px 0 6px; line-height:1.5;">
                    <b>key</b> = storage/JSON key (lowercase a-z 0-9 _) &nbsp;·&nbsp; <b>label</b> = heading injected into context &nbsp;·&nbsp; <b>description</b> = guidance for the extraction LLM.<br>
                    Renaming a key migrates existing fact sheets on Save (a backup of each sheet is taken first). Removing a category hides it — its facts stay in the sheet file untouched.
                </div>
                <div id="memPipeline_catRows"></div>
                <div style="display:flex; gap:8px; margin-top:6px;">
                    <button id="memPipeline_addCategory" class="menu_button">+ Add Category</button>
                    <button id="memPipeline_resetCategories" class="menu_button">Reset Categories to Default</button>
                </div>
            </div>

            <div style="margin-top:14px;">
                <b>Prompts</b>
                <div style="font-size:0.8em; opacity:0.7; margin:4px 0 8px; line-height:1.5;">
                    Tokens expanded at run time: <code>{{CATEGORIES}}</code>, <code>{{CATEGORY_SCHEMA}}</code>, <code>{{CATEGORY_SCHEMA_FACTS}}</code>, <code>{{CATEGORY_COUNT}}</code>, <code>{{CAP}}</code>.
                    Keep the output-format sections intact — if a load-bearing token or shape goes missing, the default block is appended automatically (with a console warning).
                    Changes apply from the next pipeline run. Press <b>Save Settings</b> to keep edits.
                </div>

                <label>Summarize prompt (per-chunk memory writing)
                    <textarea id="memPipeline_summarizePrompt" class="text_pole" rows="9" spellcheck="false"></textarea>
                </label>
                <button class="menu_button memPipeline_resetPrompt" data-target="memPipeline_summarizePrompt" data-default="summarize">Reset to default</button>

                <label>Fact extraction prompt (needs {{CATEGORIES}} + {{CATEGORY_SCHEMA}})
                    <textarea id="memPipeline_extractFactsPrompt" class="text_pole" rows="9" spellcheck="false"></textarea>
                </label>
                <button class="menu_button memPipeline_resetPrompt" data-target="memPipeline_extractFactsPrompt" data-default="extract">Reset to default</button>

                <label>Compaction prompt (needs {{CAP}} + {{CATEGORY_SCHEMA_FACTS}})
                    <textarea id="memPipeline_compactPrompt" class="text_pole" rows="9" spellcheck="false"></textarea>
                </label>
                <button class="menu_button memPipeline_resetPrompt" data-target="memPipeline_compactPrompt" data-default="compact">Reset to default</button>

                <label>Refinement prompt (needs the {"refined": [...]} output block)
                    <textarea id="memPipeline_refinePrompt" class="text_pole" rows="9" spellcheck="false"></textarea>
                </label>
                <button class="menu_button memPipeline_resetPrompt" data-target="memPipeline_refinePrompt" data-default="refine">Reset to default</button>

                <label>Consolidation prompt (merges clustered memories into one topic memory; plain-prose output)
                    <textarea id="memPipeline_consolidatePrompt" class="text_pole" rows="9" spellcheck="false"></textarea>
                </label>
                <button class="menu_button memPipeline_resetPrompt" data-target="memPipeline_consolidatePrompt" data-default="consolidate">Reset to default</button>
            </div>
        </details>

        <div id="memPipeline_status"
             style="font-size:0.85em; opacity:0.75; min-height:18px;">
            Ready.
        </div>

        <div style="margin-top:10px; font-size:0.85em; opacity:0.7;">
            <b>Tracked characters:</b>
            <div id="memPipeline_trackedList" style="margin-top:4px; line-height:1.8;"></div>
        </div>

    </div>
</div>
`;

// =============================================================================
// Fact Category Editor (Advanced drawer)
// =============================================================================

/** Escapes a string for safe use inside an HTML attribute value. */
function escapeAttr(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

/**
 * Builds one editable category row. data-original-key records the key this
 * row was rendered with, so Save can detect renames and migrate sheets.
 * New rows (Add Category) have an empty original key.
 */
function buildCategoryRow(cat = { key: '', label: '', description: '' }) {
    const row = document.createElement('div');
    row.className = 'memPipeline_catRow';
    row.dataset.originalKey = cat.key ?? '';
    row.innerHTML = `
        <input class="text_pole mp_cat_key"   placeholder="key_like_this"           value="${escapeAttr(cat.key)}" />
        <input class="text_pole mp_cat_label" placeholder="Display Label"           value="${escapeAttr(cat.label)}" />
        <input class="text_pole mp_cat_desc"  placeholder="Guidance for extraction" value="${escapeAttr(cat.description)}" />
        <button class="menu_button mp_cat_remove" title="Remove category (facts stay in the sheet file, hidden)">&#10005;</button>`;
    row.querySelector('.mp_cat_remove').addEventListener('click', () => row.remove());
    return row;
}

/** Renders the category rows from a given list (defaults to settings). */
function renderCategoryRows(categories = null) {
    const container = document.getElementById('memPipeline_catRows');
    if (!container) return;
    const cats = categories ?? getSettings().factCategories ?? DEFAULT_FACT_CATEGORIES;
    container.innerHTML = '';
    for (const cat of cats) {
        container.appendChild(buildCategoryRow(cat));
    }
}

/**
 * Reads and validates the category rows.
 * Keys are sanitised to lowercase [a-z0-9_]. Returns:
 *   cats    — the new category list
 *   renames — { oldKey: newKey } for rows whose key changed
 *   errors  — human-readable validation problems (empty/duplicate keys)
 */
function readCategoriesFromUi() {
    const rows = [...document.querySelectorAll('#memPipeline_catRows .memPipeline_catRow')];
    const cats = [];
    const renames = {};
    const seen = new Set();
    const errors = [];

    for (const row of rows) {
        const rawKey = row.querySelector('.mp_cat_key')?.value ?? '';
        const key = rawKey.trim().toLowerCase()
            .replace(/[^a-z0-9_]/g, '_')
            .replace(/_+/g, '_')
            .replace(/^_|_$/g, '');
        const label       = (row.querySelector('.mp_cat_label')?.value ?? '').trim();
        const description = (row.querySelector('.mp_cat_desc')?.value ?? '').trim();

        if (!key) { errors.push('a category row has an empty key'); continue; }
        if (seen.has(key)) { errors.push(`duplicate key "${key}"`); continue; }
        seen.add(key);

        const originalKey = row.dataset.originalKey;
        if (originalKey && originalKey !== key) {
            renames[originalKey] = key;
        }

        cats.push({ key, label: label || key, description });
    }

    return { cats, renames, errors };
}

/**
 * Migrates fact sheets for all tracked characters after category keys were
 * renamed in the settings UI: facts move from oldKey to newKey. A backup
 * snapshot of each sheet is taken BEFORE mutation. saveFactSheet is used
 * directly (not finaliseAndSave) so last_indexed timestamps are untouched
 * and incremental runs are unaffected.
 *
 * @param {object} renameMap - { oldKey: newKey }
 */
async function migrateFactSheetCategories(renameMap) {
    const settings = getSettings();
    const tracked  = settings.trackedChars ?? [];
    if (tracked.length === 0 || Object.keys(renameMap).length === 0) return;

    let migrated = 0;
    for (const avatarUrl of tracked) {
        try {
            const sheet = await loadFactSheet(avatarUrl);
            if (!sheet?.facts) continue;

            const touches = Object.keys(renameMap).some(oldKey => oldKey in sheet.facts);
            if (!touches) continue;

            // Snapshot the pre-migration state first. Never blocks.
            await backupFactSheet(avatarUrl, JSON.parse(JSON.stringify(sheet)));

            for (const [oldKey, newKey] of Object.entries(renameMap)) {
                if (!(oldKey in sheet.facts)) continue;
                const moving   = Array.isArray(sheet.facts[oldKey]) ? sheet.facts[oldKey] : [];
                const existing = Array.isArray(sheet.facts[newKey]) ? sheet.facts[newKey] : [];
                sheet.facts[newKey] = [...moving, ...existing];
                delete sheet.facts[oldKey];
                console.log(`${LOG_PREFIX} Migrated ${moving.length} fact(s): "${oldKey}" → "${newKey}" for ${avatarUrl}`);
            }

            await saveFactSheet(avatarUrl, sheet);
            migrated++;
        } catch (err) {
            console.warn(`${LOG_PREFIX} Category migration failed for "${avatarUrl}":`, err);
        }
    }

    if (migrated > 0) {
        setStatus(`Settings saved. Category keys migrated in ${migrated} fact sheet(s) — backups taken.`);
    }
}

/**
 * Populates the UI fields from current settings.
 */
function syncUiFromSettings() {
    const s = getSettings();
    const get = id => document.getElementById(id);

    get('memPipeline_enabled').checked              = s.enabled;
    get('memPipeline_refineSummaries').checked      = s.refineSummaries;
    get('memPipeline_llmPort').value                = s.llmPort;
    get('memPipeline_embeddingPort').value          = s.embeddingPort;
    get('memPipeline_qdrantPort').value             = s.qdrantPort;
    get('memPipeline_vectorSize').value             = s.vectorSize;
    get('memPipeline_embeddingModel').value         = s.embeddingModel;
    get('memPipeline_embeddingApi').value           = s.embeddingApi;
    get('memPipeline_qdrantUrl').value              = s.qdrantUrl;
    get('memPipeline_embeddingUrl').value           = s.embeddingUrl;
    get('memPipeline_sheetPosition').value          = s.sheetPosition;
    get('memPipeline_compactEveryNSessions').value  = s.compactEveryNSessions;
    get('memPipeline_chunkSize').value              = s.chunkSize;
    get('memPipeline_chunkGapHours').value          = s.chunkGapHours;
    get('memPipeline_scheduledTime').value          = s.scheduledTime;
    get('memPipeline_factsPerCategoryCap').value    = s.factsPerCategoryCap;
    get('memPipeline_backupsRetention').value       = s.backupsRetention;
    get('memPipeline_shelfEnabled').checked         = s.shelfEnabled;
    get('memPipeline_shelfServer').value            = s.shelfServer;

    // Recall (read path)
    get('memPipeline_recallEnabled').checked             = s.recallEnabled;
    get('memPipeline_recallK').value                     = s.recallK;
    get('memPipeline_recallThreshold').value             = s.recallThreshold;
    get('memPipeline_recallPosition').value              = s.recallPosition;
    get('memPipeline_recallRetainRecent').value          = s.recallRetainRecent;
    get('memPipeline_recallNeighborWindowDays').value    = s.recallNeighborWindowDays;
    get('memPipeline_recallNeighborsPerMatch').value     = s.recallNeighborsPerMatch;
    get('memPipeline_recallNeighborsTotal').value        = s.recallNeighborsTotal;
    get('memPipeline_recallHeader').value                = s.recallHeader;

    get('memPipeline_consolidationEnabled').checked     = s.consolidationEnabled;
    get('memPipeline_consolidationThreshold').value      = s.consolidationThreshold;
    get('memPipeline_consolidationMinCluster').value     = s.consolidationMinCluster;
    get('memPipeline_consolidationMaxClusters').value    = s.consolidationMaxClusters;

    // Advanced: prompt textareas + category editor rows.
    get('memPipeline_summarizePrompt').value    = s.summarizePrompt    ?? DEFAULT_SUMMARIZE_PROMPT;
    get('memPipeline_extractFactsPrompt').value = s.extractFactsPrompt ?? DEFAULT_EXTRACT_FACTS_PROMPT;
    get('memPipeline_compactPrompt').value      = s.compactPrompt      ?? DEFAULT_COMPACT_PROMPT;
    get('memPipeline_refinePrompt').value       = s.refinePrompt       ?? DEFAULT_REFINE_PROMPT;
    get('memPipeline_consolidatePrompt').value  = s.consolidatePrompt  ?? DEFAULT_CONSOLIDATE_PROMPT;
    renderCategoryRows();

    // Render tracked characters list.
    const listEl = get('memPipeline_trackedList');
    if (listEl) {
        listEl.textContent = s.trackedChars.length > 0
            ? s.trackedChars.join(', ')
            : 'None yet — run the pipeline for a character to track it.';
    }
}

/**
 * Reads UI fields and persists them to extension_settings.
 */
async function syncSettingsFromUi() {
    const s   = getSettings();
    const get = id => document.getElementById(id);

    // --- Fact categories: validate BEFORE saving anything, so a broken
    //     category edit can't corrupt the pipeline's source of truth. ---
    const { cats, renames, errors } = readCategoriesFromUi();
    if (errors.length > 0) {
        setStatus(`Category error: ${errors.join('; ')} — settings NOT saved.`);
        return;
    }
    if (cats.length === 0) {
        setStatus('At least one fact category is required — settings NOT saved.');
        return;
    }
    if (Object.keys(renames).length > 0) {
        const renameList = Object.entries(renames).map(([o, n]) => `  ${o} → ${n}`).join('\n');
        const ok = confirm(
            `You renamed category key(s):\n${renameList}\n\n` +
            `Existing fact sheets for all tracked characters will be migrated ` +
            `(facts moved to the new key). A backup of each sheet is taken first.\n\nContinue?`
        );
        if (!ok) {
            setStatus('Save cancelled — category rename not applied.');
            return;
        }
    }

    s.enabled              = get('memPipeline_enabled').checked;
    s.refineSummaries      = get('memPipeline_refineSummaries').checked;
    s.llmPort              = parseInt(get('memPipeline_llmPort').value, 10)              || DEFAULT_SETTINGS.llmPort;
    s.embeddingPort        = parseInt(get('memPipeline_embeddingPort').value, 10)        || DEFAULT_SETTINGS.embeddingPort;
    s.qdrantPort           = parseInt(get('memPipeline_qdrantPort').value, 10)           || DEFAULT_SETTINGS.qdrantPort;
    s.vectorSize           = parseInt(get('memPipeline_vectorSize').value, 10)           || DEFAULT_SETTINGS.vectorSize;
    s.embeddingModel       = (get('memPipeline_embeddingModel').value || '').trim() || DEFAULT_READER_SETTINGS.embeddingModel;
    s.embeddingApi         = ((get('memPipeline_embeddingApi').value || '').trim().toLowerCase() === 'openai') ? 'openai' : 'ollama';
    s.qdrantUrl            = (get('memPipeline_qdrantUrl').value || '').trim().replace(/\/+$/, '');
    s.embeddingUrl         = (get('memPipeline_embeddingUrl').value || '').trim().replace(/\/+$/, '');
    s.chunkSize            = parseInt(get('memPipeline_chunkSize').value, 10)            || DEFAULT_SETTINGS.chunkSize;

    s.shelfEnabled         = get('memPipeline_shelfEnabled').checked;
    s.shelfServer          = get('memPipeline_shelfServer').value.trim();

    // --- Recall (read path) knobs. Zero is legitimate for several of
    //     these (position 0, retain 0, window 0 = expansion off), so use
    //     the Number+isFinite idiom rather than ||-fallbacks. ---
    s.recallEnabled = get('memPipeline_recallEnabled').checked;

    const readIntMin0 = (id, fallback) => {
        const raw = parseInt(get(id)?.value, 10);
        return Number.isFinite(raw) && raw >= 0 ? raw : fallback;
    };
    s.recallK                     = parseInt(get('memPipeline_recallK').value, 10) || DEFAULT_READER_SETTINGS.recallK;
    s.sheetPosition               = readIntMin0('memPipeline_sheetPosition',           DEFAULT_SETTINGS.sheetPosition);
    s.compactEveryNSessions       = readIntMin0('memPipeline_compactEveryNSessions',   DEFAULT_SETTINGS.compactEveryNSessions);
    s.recallPosition              = readIntMin0('memPipeline_recallPosition',           DEFAULT_READER_SETTINGS.recallPosition);
    s.recallRetainRecent          = readIntMin0('memPipeline_recallRetainRecent',       DEFAULT_READER_SETTINGS.recallRetainRecent);
    s.recallNeighborWindowDays    = readIntMin0('memPipeline_recallNeighborWindowDays', DEFAULT_READER_SETTINGS.recallNeighborWindowDays);
    s.recallNeighborsPerMatch     = readIntMin0('memPipeline_recallNeighborsPerMatch',  DEFAULT_READER_SETTINGS.recallNeighborsPerMatch);
    s.recallNeighborsTotal        = readIntMin0('memPipeline_recallNeighborsTotal',     DEFAULT_READER_SETTINGS.recallNeighborsTotal);

    const recallThrRaw = parseFloat(get('memPipeline_recallThreshold').value);
    s.recallThreshold = (Number.isFinite(recallThrRaw) && recallThrRaw >= 0 && recallThrRaw <= 1)
        ? recallThrRaw
        : DEFAULT_READER_SETTINGS.recallThreshold;

    // (Self-contained read — the shared readPrompt helper is declared
    //  further down this function and isn't in scope yet at this point.)
    const recallHeaderRaw = get('memPipeline_recallHeader')?.value ?? '';
    s.recallHeader = recallHeaderRaw.trim().length > 0
        ? recallHeaderRaw
        : DEFAULT_READER_SETTINGS.recallHeader;

    // chunkGapHours: 0 is a legitimate value (= disabled), so use the
    // Number+isFinite idiom like backupsRetention, not the ||-fallback.
    const gapRaw = parseFloat(get('memPipeline_chunkGapHours').value);
    s.chunkGapHours = Number.isFinite(gapRaw) && gapRaw >= 0
        ? gapRaw
        : DEFAULT_SETTINGS.chunkGapHours;
    s.factsPerCategoryCap  = parseInt(get('memPipeline_factsPerCategoryCap').value, 10)  || DEFAULT_SETTINGS.factsPerCategoryCap;

    // backupsRetention deliberately uses Number+isFinite, not the
    // ||-fallback idiom, because 0 is a legitimate value (= unlimited)
    // and falsy-coercion would silently rewrite it to the default.
    const retentionRaw = parseInt(get('memPipeline_backupsRetention').value, 10);
    s.backupsRetention = Number.isFinite(retentionRaw) && retentionRaw >= 0
        ? retentionRaw
        : DEFAULT_SETTINGS.backupsRetention;

    const rawTime = get('memPipeline_scheduledTime').value.trim();
    s.scheduledTime = /^\d{2}:\d{2}$/.test(rawTime) ? rawTime : DEFAULT_SETTINGS.scheduledTime;

    // --- Prompts: empty textarea = revert to the shipped default. ---
    const readPrompt = (id, fallback) => {
        const v = get(id)?.value ?? '';
        return v.trim().length > 0 ? v : fallback;
    };
    s.summarizePrompt    = readPrompt('memPipeline_summarizePrompt',    DEFAULT_SUMMARIZE_PROMPT);
    s.extractFactsPrompt = readPrompt('memPipeline_extractFactsPrompt', DEFAULT_EXTRACT_FACTS_PROMPT);
    s.compactPrompt      = readPrompt('memPipeline_compactPrompt',      DEFAULT_COMPACT_PROMPT);
    s.refinePrompt       = readPrompt('memPipeline_refinePrompt',       DEFAULT_REFINE_PROMPT);
    s.consolidatePrompt  = readPrompt('memPipeline_consolidatePrompt',  DEFAULT_CONSOLIDATE_PROMPT);

    // Consolidation tuning knobs.
    s.consolidationEnabled = get('memPipeline_consolidationEnabled').checked;
    const thrRaw = parseFloat(get('memPipeline_consolidationThreshold').value);
    s.consolidationThreshold = (Number.isFinite(thrRaw) && thrRaw >= 0.5 && thrRaw <= 0.99)
        ? thrRaw
        : DEFAULT_SETTINGS.consolidationThreshold;
    s.consolidationMinCluster  = parseInt(get('memPipeline_consolidationMinCluster').value, 10)  || DEFAULT_SETTINGS.consolidationMinCluster;
    s.consolidationMaxClusters = parseInt(get('memPipeline_consolidationMaxClusters').value, 10) || DEFAULT_SETTINGS.consolidationMaxClusters;

    // --- Categories: commit, then migrate any renamed keys. ---
    s.factCategories = cats;

    saveSettings();
    setStatus('Settings saved.');

    if (Object.keys(renames).length > 0) {
        await migrateFactSheetCategories(renames);
    }

    // Re-render rows so data-original-key reflects the just-saved keys
    // (otherwise a second Save would re-detect the same renames).
    renderCategoryRows();

    // Restart the scheduler with the new time.
    scheduleNextRun(s.scheduledTime);
    console.log(`${LOG_PREFIX} Settings updated.`);
}

/**
 * Attaches event listeners to all UI elements.
 */
function bindUiEvents() {
    document.getElementById('memPipeline_saveSettings')
        ?.addEventListener('click', () => { syncSettingsFromUi(); });

    // --- Advanced: category editor buttons ---
    document.getElementById('memPipeline_addCategory')
        ?.addEventListener('click', () => {
            document.getElementById('memPipeline_catRows')
                ?.appendChild(buildCategoryRow());
        });

    document.getElementById('memPipeline_resetCategories')
        ?.addEventListener('click', () => {
            const ok = confirm(
                'Reset the category list to the shipped defaults?\n\n' +
                'This only changes the EDITOR — nothing is saved until you press ' +
                'Save Settings. Facts stored under non-default keys stay in the ' +
                'sheet files (hidden until their key exists again or is migrated).'
            );
            if (!ok) return;
            const currentKeys = new Set((getSettings().factCategories ?? []).map(c => c.key));
            // Rows for default keys that already exist keep their original-key
            // marker (no rename); keys new to this config are marked as new.
            const rows = DEFAULT_FACT_CATEGORIES.map(c =>
                currentKeys.has(c.key) ? c : { ...c, key: c.key });
            renderCategoryRows(rows);
            // Mark rows whose key is NOT currently saved as brand-new so a
            // Save doesn't try to "rename" from a nonexistent key.
            document.querySelectorAll('#memPipeline_catRows .memPipeline_catRow').forEach(row => {
                if (!currentKeys.has(row.dataset.originalKey)) row.dataset.originalKey = '';
            });
            setStatus('Categories reset in editor — press Save Settings to keep.');
        });

    // --- Advanced: per-prompt reset buttons ---
    const PROMPT_DEFAULTS = {
        summarize:   DEFAULT_SUMMARIZE_PROMPT,
        extract:     DEFAULT_EXTRACT_FACTS_PROMPT,
        compact:     DEFAULT_COMPACT_PROMPT,
        refine:      DEFAULT_REFINE_PROMPT,
        consolidate: DEFAULT_CONSOLIDATE_PROMPT,
    };
    document.querySelectorAll('.memPipeline_resetPrompt').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = document.getElementById(btn.dataset.target);
            const def    = PROMPT_DEFAULTS[btn.dataset.default];
            if (target && def !== undefined) {
                target.value = def;
                setStatus('Prompt reset in editor — press Save Settings to keep.');
            }
        });
    });

    // Resolves the currently-open character, or null (with a status message)
    // if none is selectable. Shared by both run buttons.
    const getCurrentCharacter = () => {
        const context = getContext();
        if (!context.characterId && context.characterId !== 0) {
            setStatus('No character selected. Open a character chat first.');
            return null;
        }
        const char      = context.characters[context.characterId];
        const avatarUrl = char?.avatar;
        const charName  = char?.name;
        if (!avatarUrl || !charName) {
            setStatus('Could not read character info. Try reloading.');
            return null;
        }
        return { avatarUrl, charName };
    };

    // Disable both run buttons during any run so they can't overlap (two
    // concurrent pipeline runs on the same character would race on the
    // fact sheet and the Qdrant collection).
    const setRunButtonsDisabled = (disabled) => {
        const a = document.getElementById('memPipeline_runNow');
        const b = document.getElementById('memPipeline_forceReindex');
        const c = document.getElementById('memPipeline_rebuildConsolidated');
        if (a) a.disabled = disabled;
        if (b) b.disabled = disabled;
        if (c) c.disabled = disabled;
    };

    document.getElementById('memPipeline_runNow')
        ?.addEventListener('click', async () => {
            const target = getCurrentCharacter();
            if (!target) return;

            setRunButtonsDisabled(true);
            await runPipelineForCharacter(target.charName, target.avatarUrl);
            setRunButtonsDisabled(false);

            // Refresh the tracked list in the UI.
            syncUiFromSettings();
        });

    document.getElementById('memPipeline_forceReindex')
        ?.addEventListener('click', async () => {
            const target = getCurrentCharacter();
            if (!target) return;

            const ok = confirm(
                `Force a FULL reindex for "${target.charName}"?\n\n` +
                `This reprocesses every chat session from scratch and completely ` +
                `rebuilds the Qdrant collection. It can take a while and makes many ` +
                `LLM calls.\n\n` +
                `Your fact sheet is preserved — it gets re-merged and compacted, not reset.`
            );
            if (!ok) return;

            setRunButtonsDisabled(true);
            await runPipelineForCharacter(target.charName, target.avatarUrl, true);
            setRunButtonsDisabled(false);

            syncUiFromSettings();
        });

    document.getElementById('memPipeline_rebuildConsolidated')
        ?.addEventListener('click', async () => {
            const target = getCurrentCharacter();
            if (!target) return;

            const settings = getSettings();
            setRunButtonsDisabled(true);
            setStatus(`Rebuilding consolidated memories for ${target.charName}...`);
            try {
                const cRes = await runConsolidation(target.charName, {
                    qdrantPort:     settings.qdrantPort,
                    embeddingPort:  settings.embeddingPort,
                    embeddingModel: settings.embeddingModel,
                    embeddingUrl:   settings.embeddingUrl,
                    embeddingApi:   settings.embeddingApi,
                    llmPort:        settings.llmPort,
                    threshold:      settings.consolidationThreshold,
                    minCluster:     settings.consolidationMinCluster,
                    maxClusters:    settings.consolidationMaxClusters,
                    promptOverride: settings.consolidatePrompt,
                });
                setStatus(`Consolidation done — ${cRes.indexed} topic memor${cRes.indexed === 1 ? 'y' : 'ies'} from ${cRes.clustersQualified} cluster(s) (${cRes.basePoints} base points).`);
            } catch (err) {
                console.error(`${LOG_PREFIX} Manual consolidation failed:`, err);
                setStatus(`Consolidation error: ${err.message}`);
            }
            setRunButtonsDisabled(false);
        });

    // --- Recall (read path): connection tests, count display, preview ---

    document.getElementById('memPipeline_testQdrant')
        ?.addEventListener('click', async () => {
            setStatus('Testing Qdrant connection...');
            const res = await testQdrantConnection(getSettings().qdrantPort, getSettings().qdrantUrl);
            setStatus(`${res.ok ? '✓' : '✗'} ${res.detail}`);
        });

    document.getElementById('memPipeline_testEmbedding')
        ?.addEventListener('click', async () => {
            setStatus('Testing embedding endpoint...');
            const s2 = getSettings();
            const res = await testEmbeddingConnection(s2.embeddingPort, s2.embeddingModel, s2.embeddingUrl, s2.embeddingApi);
            setStatus(`${res.ok ? '✓' : '✗'} ${res.detail}`);
        });

    const refreshMemoryCount = async () => {
        const countEl = document.getElementById('memPipeline_memoryCount');
        if (!countEl) return;
        // Quiet character resolution — no status-line nagging on page load.
        const context = getContext();
        const char    = context.characters?.[context.characterId];
        const target  = char?.name ? { charName: char.name } : null;
        if (!target) { countEl.textContent = '—'; return; }
        try {
            const n = await countMemories(target.charName, getSettings().qdrantPort, getSettings().qdrantUrl);
            countEl.textContent = (n === null)
                ? 'no collection yet'
                : String(n);
        } catch {
            countEl.textContent = 'unreachable';
        }
    };
    document.getElementById('memPipeline_refreshCount')
        ?.addEventListener('click', refreshMemoryCount);
    // Populate once on load, quietly.
    refreshMemoryCount();

    const runPreview = async (wide) => {
        const target = getCurrentCharacter();
        if (!target) return;

        const query = document.getElementById('memPipeline_previewQuery')?.value?.trim();
        if (!query) { setStatus('Type a test query first.'); return; }

        const out = document.getElementById('memPipeline_previewOutput');
        setStatus(wide ? 'Running WIDE preview (top 20, no threshold)...' : 'Running recall preview...');
        try {
            const { report } = await previewRecall(query, target.charName, getSettings(), wide);
            if (out) { out.textContent = report; out.style.display = 'block'; }
            setStatus('Preview complete — nothing was sent to the model.');
        } catch (err) {
            if (out) { out.textContent = `Preview failed: ${err.message}`; out.style.display = 'block'; }
            setStatus(`Preview error: ${err.message}`);
        }
    };
    document.getElementById('memPipeline_previewRecall')
        ?.addEventListener('click', () => runPreview(false));
    document.getElementById('memPipeline_previewWide')
        ?.addEventListener('click', () => runPreview(true));

    // --- Fact-sheet editor: quick corrections, backup before every save ---
    let sheetBeingEdited = null;   // { avatarUrl, sheet } while the editor is open
    const sheetWrap = () => document.getElementById('memPipeline_sheetEditorWrap');
    const sheetArea = () => document.getElementById('memPipeline_sheetEditor');

    document.getElementById('memPipeline_editSheet')
        ?.addEventListener('click', async () => {
            const target = getCurrentCharacter();
            if (!target) return;
            try {
                const sheet = await loadFactSheet(target.avatarUrl);
                if (!sheet) { setStatus('No fact sheet yet for this character — run the pipeline first.'); return; }
                sheetBeingEdited = { avatarUrl: target.avatarUrl, sheet };
                sheetArea().value = JSON.stringify(sheet.facts ?? {}, null, 2);
                sheetWrap().style.display = 'block';
                setStatus(`Editing fact sheet for ${target.charName}. Save takes a backup first.`);
            } catch (err) {
                setStatus(`Could not load fact sheet: ${err.message}`);
            }
        });

    document.getElementById('memPipeline_sheetCancel')
        ?.addEventListener('click', () => {
            sheetBeingEdited = null;
            sheetWrap().style.display = 'none';
            setStatus('Fact-sheet edit cancelled — nothing changed.');
        });

    document.getElementById('memPipeline_sheetSave')
        ?.addEventListener('click', async () => {
            if (!sheetBeingEdited) return;
            let parsed;
            try {
                parsed = JSON.parse(sheetArea().value);
            } catch (err) {
                setStatus(`Not valid JSON — nothing saved. (${err.message})`); return;
            }
            // Shape check: object of arrays of {fact: string, added: string}.
            if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
                setStatus('Top level must be an object of categories — nothing saved.'); return;
            }
            for (const [cat, arr] of Object.entries(parsed)) {
                if (!Array.isArray(arr)) { setStatus(`Category "${cat}" must be an array — nothing saved.`); return; }
                for (const e of arr) {
                    if (typeof e?.fact !== 'string' || typeof e?.added !== 'string') {
                        setStatus(`Every entry in "${cat}" needs string "fact" and "added" — nothing saved.`); return;
                    }
                }
            }
            try {
                const { avatarUrl, sheet } = sheetBeingEdited;
                await backupFactSheet(avatarUrl, JSON.parse(JSON.stringify(sheet)));
                sheet.facts = parsed;
                await saveFactSheet(avatarUrl, sheet);
                sheetBeingEdited = null;
                sheetWrap().style.display = 'none';
                setStatus('Fact sheet saved (backup of the previous version taken). Injects from the next message.');
            } catch (err) {
                setStatus(`Save failed: ${err.message}`);
            }
        });
}

// =============================================================================
// Extension Init
// =============================================================================

/**
 * Called by SillyTavern when the extension loads.
 * Registers the settings panel, binds events, and starts the scheduler.
 */
async function init() {
    console.log(`${LOG_PREFIX} Initialising...`);

    // Ensure settings exist with defaults.
    getSettings();

    // Use jQuery (ST global) to inject the drawer into the Extensions tab.
    // #extensions_settings2 is the confirmed target used by other ST extensions.
    const $ = window.$;
    $('#extensions_settings2').append(SETTINGS_HTML);

    // Wire up the collapse/expand chevron behaviour ST uses for all drawers.
    if (typeof window.applyInlineDrawerListeners === 'function') {
        window.applyInlineDrawerListeners();
    }

    // Populate fields and attach listeners.
    syncUiFromSettings();
    bindUiEvents();

    // Start the daily scheduler.
    const settings = getSettings();
    if (settings.enabled) {
        scheduleNextRun(settings.scheduledTime);
    }

    console.log(`${LOG_PREFIX} Ready. Scheduled time: ${settings.scheduledTime}`);
}

// =============================================================================
// Chat-time Injection (generate_interceptor)
// =============================================================================
//
// Registered via manifest.json's "generate_interceptor": "memoryPipelineInterceptor".
// ST calls this hook on every generation, before sending the prompt to the
// model. Both halves of the Lattice's per-message injection happen here:
//
//   1. READ PATH (reader.js): the outgoing message is embedded, relevant
//      memories (and their temporal neighbors) are retrieved from Qdrant,
//      and the rendered block is spliced in at the configured depth. Shelf
//      entries (kind 'hands'/'gift') live in the same collection, so the
//      hands' offline writing surfaces through the same recall.
//   2. FACT SHEET: the current character's fact sheet is prepended as a
//      system message at the very top — identity and current state are
//      never a retrieval gamble.
//
// Performance note: we load the fact sheet from disk on every message. This
// is a single small JSON file fetched from a local endpoint, so the latency
// is negligible compared to LLM generation. If it ever becomes a bottleneck,
// add a per-character cache invalidated whenever finaliseAndSave runs.
// =============================================================================

async function memoryPipelineInterceptor(chat, _contextSize, _abort, type) {
    const settings = getSettings();
    if (!settings.enabled) return;

    const context = getContext();
    const characters = context.characters ?? [];
    const character = characters[context.characterId];
    if (!character?.avatar) return; // group chat, or no character selected

    // --- READ PATH: recalled memories, spliced at the configured depth. ---
    // Runs FIRST so the fact sheet's unshift below always lands on top.
    // Best-effort: any recall failure degrades to fact-sheet-only.
    try {
        const injectedTokens = await injectRecalledMemories(
            chat, character.name, settings, type);
        if (injectedTokens > 0) {
            console.log(`${LOG_PREFIX} Recall injected (~${injectedTokens} tokens).`);
        }
    } catch (err) {
        console.warn(`${LOG_PREFIX} Recall failed (non-blocking):`, err);
    }

    // --- Fact sheet: identity and current state. ---
    try {
        // Double-call guard (same class as the reader's and state-bridge's):
        // a second interceptor invocation must not unshift the sheet twice.
        if (chat.some(m => m && m[SHEET_MARKER])) return;

        const sheet = await loadFactSheet(character.avatar);
        if (!sheet) return; // no fact sheet yet — pipeline hasn't run for this character

        const formatted = formatFactSheetForContext(sheet, getSettings().factCategories);
        if (!formatted) return; // sheet exists but is empty

        const entry = {
            is_user:   false,
            is_system: true,
            name:      'System',
            mes:       formatted,
            send_date: new Date().toISOString(),
            [SHEET_MARKER]: true,
        };
        const pos = Math.max(0, settings.sheetPosition ?? 0);
        if (pos === 0) {
            // Classic: pinned at the very top, ahead of every other injection —
            // durable foundational context.
            chat.unshift(entry);
        } else {
            // N messages from the end, nearer the turn being generated.
            const at = Math.max(0, chat.length - Math.min(chat.length, pos));
            chat.splice(at, 0, entry);
            console.log(`${LOG_PREFIX} fact sheet spliced at index ${at} of ${chat.length - 1} (sheetPosition ${pos}).`);
        }
    } catch (err) {
        // An injection failure must NEVER block generation. Log and continue.
        console.warn(`${LOG_PREFIX} Fact sheet injection failed:`, err);
    }
}

// ST resolves the interceptor by looking up the name in the global scope,
// so we must attach our function there explicitly (ES module declarations
// don't leak to globalThis on their own).
globalThis.memoryPipelineInterceptor = memoryPipelineInterceptor;


// ST calls init() automatically via the manifest's js entry point.
init().catch(err => console.error(`${LOG_PREFIX} Init failed:`, err));

