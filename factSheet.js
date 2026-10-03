// =============================================================================
// factSheet.js — Memory Pipeline: Fact Sheet Persistence
// =============================================================================
// Responsibilities:
//   1. Derive a consistent, safe filename for each character's fact sheet
//   2. Save a fact sheet to ST's user files directory via /api/files/upload
//   3. Load a fact sheet back from ST's user files directory via GET
//   4. Store and retrieve the last_indexed timestamp (lives inside the sheet)
//   5. Provide a formatted string of the fact sheet for context injection
//   6. Take timestamped pre-merge backup snapshots of the fact sheet (v2)
// =============================================================================

'use strict';

// SillyTavern attaches a CSRF token to every /api/... request. Using
// getRequestHeaders() returns { Content-Type, X-CSRF-Token } so our calls
// authenticate properly — otherwise ST responds with HTTP 403.
import { getRequestHeaders } from '../../../../script.js';
import { DEFAULT_FACT_CATEGORIES } from './summarizer.js';

const LOG_PREFIX = '[MemoryPipeline/FactSheet]';

// Prefix for all fact sheet filenames so they're easy to identify in the
// user files directory and don't collide with other extension files.
const FILE_PREFIX = 'memPipeline_';

// Backups go to the same user files directory but with their own prefix so
// they don't get mistaken for live fact sheets, and so any future "list all
// my backups" tooling can glob on this prefix. ST doesn't allow slashes in
// file names (validateAssetFileName rejects them), so we can't put backups
// in a true subfolder — the prefix is the next best thing.
const BACKUP_PREFIX = 'memPipeline_backup_';


// =============================================================================
// SECTION 1: Filename Derivation
// =============================================================================

/**
 * Derives a safe, consistent filename for a character's fact sheet.
 *
 * Input:  "Fin.png"
 * Output: "memPipeline_Fin.json"
 *
 * Input:  "My Character (v2).png"
 * Output: "memPipeline_My_Character__v2_.json"
 *
 * The validateAssetFileName check in ST requires: [a-zA-Z0-9_\-.]
 * We replace anything outside that set with underscores to be safe.
 *
 * @param {string} avatarUrl  - Character avatar filename e.g. "Fin.png"
 * @returns {string}          - Safe filename e.g. "memPipeline_Fin.json"
 */
function deriveFileName(avatarUrl) {
    // Strip the image extension (.png, .jpg, etc.)
    const baseName = avatarUrl.replace(/\.[^/.]+$/, '');
    // Replace any character not allowed by ST's validator with underscore.
    const safeName = baseName.replace(/[^a-zA-Z0-9_\-.]/g, '_');
    return `${FILE_PREFIX}${safeName}.json`;
}


// =============================================================================
// SECTION 2: Save
// =============================================================================

/**
 * Saves a fact sheet to ST's user files directory.
 *
 * Uses POST /api/files/upload which expects:
 *   { name: string, data: string (base64) }
 *
 * The endpoint writes to request.user.directories.files and returns:
 *   { path: string }  — the client-relative path for reading back
 *
 * We store the returned path inside the fact sheet itself so we always
 * know where to read it from, without having to rederive or guess.
 *
 * @param {string} avatarUrl   - Character avatar filename
 * @param {object} sheet       - The full fact sheet object from mergeFactSheet()
 * @returns {Promise<string>}  - The client-relative path ST assigned the file
 */
async function saveFactSheet(avatarUrl, sheet) {
    const fileName = deriveFileName(avatarUrl);

    // Encode the JSON as base64 — required by the upload endpoint.
    const json = JSON.stringify(sheet, null, 2);
    const base64 = btoa(unescape(encodeURIComponent(json)));

    const response = await fetch('/api/files/upload', {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({
            name: fileName,
            data: base64,
        }),
    });

    if (!response.ok) {
        const msg = await response.text();
        throw new Error(`${LOG_PREFIX} Failed to save fact sheet for "${avatarUrl}": ${msg}`);
    }

    const result = await response.json();
    const filePath = result.path;

    console.log(`${LOG_PREFIX} Saved fact sheet for "${avatarUrl}" → ${filePath}`);
    return filePath;
}


// =============================================================================
// SECTION 3: Load
// =============================================================================

/**
 * Loads a character's fact sheet from ST's user files directory.
 *
 * ST's upload endpoint returns a client-relative path (e.g. "files/memPipeline_Fin.json").
 * ST serves user files as static assets, so a GET to that path retrieves the file.
 *
 * On first run (no fact sheet exists yet), returns null so the pipeline knows
 * to initialise a fresh sheet.
 *
 * @param {string} avatarUrl  - Character avatar filename
 * @returns {Promise<object|null>}  - Parsed fact sheet, or null if not found
 */
async function loadFactSheet(avatarUrl) {
    const fileName = deriveFileName(avatarUrl);

    // Step 1: Verify the file exists before trying to fetch it.
    // /api/files/verify takes { urls: string[] } and returns { [url]: boolean }
    //
    // ST stores per-user files at data/<user>/user/files/<name>. The upload
    // endpoint returns a client-relative path of "/user/files/<name>" and
    // serves them at that same URL. We must include the "user/" segment
    // here — without it the verify call returns false (the file looks
    // missing) and we silently skip injection even when the sheet exists.
    const expectedPath = `user/files/${fileName}`;

    const verifyResponse = await fetch('/api/files/verify', {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({ urls: [expectedPath] }),
    });

    if (!verifyResponse.ok) {
        console.warn(`${LOG_PREFIX} Could not verify fact sheet existence for "${avatarUrl}".`);
        return null;
    }

    const verified = await verifyResponse.json();

    if (!verified[expectedPath]) {
        // No fact sheet yet — first run for this character. 
        console.log(`${LOG_PREFIX} No fact sheet found for "${avatarUrl}" — will create on first pipeline run.`);
        return null;
    }

    // Step 2: Fetch the file. ST serves user files from /files/ as static assets.
    const getResponse = await fetch(`/${expectedPath}`, {
        method: 'GET',
        headers: { 'Cache-Control': 'no-cache' },
    });

    if (!getResponse.ok) {
        console.warn(`${LOG_PREFIX} Fact sheet exists but could not be read for "${avatarUrl}". Status: ${getResponse.status}`);
        return null;
    }

    try {
        const sheet = await getResponse.json();
        console.log(`${LOG_PREFIX} Loaded fact sheet for "${avatarUrl}" (last updated: ${sheet.last_updated ?? 'unknown'})`);
        return sheet;
    } catch (err) {
        console.error(`${LOG_PREFIX} Fact sheet for "${avatarUrl}" is corrupt or unparseable. Will reinitialise.`);
        return null;
    }
}


// =============================================================================
// SECTION 4: Timestamp Management
// =============================================================================

/**
 * Returns the last_indexed Unix timestamp stored in a fact sheet.
 * Used by the pipeline to filter which chats are new since the last run.
 *
 * Returns 0 if the sheet is null or has no timestamp, which causes
 * loadChatsForCharacter to process ALL sessions (correct for first run).
 *
 * @param {object|null} sheet
 * @returns {number} - Unix timestamp in ms, or 0
 */
function getLastIndexed(sheet) {
    return sheet?.last_indexed_ms ?? 0;
}

/**
 * Stamps the current time as last_indexed_ms onto the fact sheet.
 * Called at the end of a successful pipeline run, before saving.
 *
 * @param {object} sheet
 * @returns {object} - The updated sheet (mutated in place and returned)
 */
function stampLastIndexed(sheet) {
    sheet.last_indexed_ms = Date.now();
    sheet.last_indexed_date = new Date().toISOString().split('T')[0];
    return sheet;
}


// =============================================================================
// SECTION 5: Context Injection Formatting
// =============================================================================

/**
 * Formats the fact sheet into a clean, readable string for injection into
 * the ST context window (pinned at the top, above Qdrant summaries).
 *
 * The output is intentionally plain and terse — this goes into the model's
 * context, not a UI. No markdown headers, just labelled category blocks.
 *
 * Example output:
 *
 *   [Character Memory: Fin]
 *   Names & Relationships:
 *   - User's name is Alex. (2026-01-15)
 *   - Fin considers Alex a close friend. (2026-02-20)
 *
 *   Preferences & Dislikes:
 *   - Alex dislikes coffee. (2026-03-01)
 *   ...
 *
 * @param {object|null} sheet       - Fact sheet from loadFactSheet()
 * @param {object[]}    [categories] - Active category list from settings
 *                                     ({ key, label, description }); falls
 *                                     back to the shipped defaults.
 * @returns {string}           - Formatted string ready for context injection,
 *                               or empty string if sheet is null/empty
 */
function formatFactSheetForContext(sheet, categories = null) {
    if (!sheet || !sheet.facts) return '';

    const cats = (Array.isArray(categories) && categories.length > 0)
        ? categories
        : DEFAULT_FACT_CATEGORIES;

    const lines = [`[Character Memory: ${sheet.character ?? 'Unknown'}]`];
    let hasAnyFacts = false;

    for (const cat of cats) {
        const facts = sheet.facts[cat.key];
        if (!Array.isArray(facts) || facts.length === 0) continue;

        hasAnyFacts = true;
        lines.push(`${cat.label ?? cat.key}:`);

        for (const entry of facts) {
            // entry is { fact: string, added: string (ISO date) }
            lines.push(`- ${entry.fact} (${entry.added})`);
        }

        lines.push(''); // blank line between categories
    }

    if (!hasAnyFacts) return '';

    return lines.join('\n').trim();
}


// =============================================================================
// SECTION 6: Full Save Lifecycle Helper
// =============================================================================

/**
 * Convenience function called at the end of a successful pipeline run.
 * Stamps the timestamp, saves the sheet, and returns the saved sheet.
 *
 * Usage in the pipeline orchestrator (index.js):
 *   const saved = await finaliseAndSave(avatarUrl, updatedSheet);
 *
 * @param {string} avatarUrl
 * @param {object} sheet      - Updated sheet from runPipeline()
 * @returns {Promise<object>} - The stamped, saved sheet
 */
async function finaliseAndSave(avatarUrl, sheet) {
    const stamped = stampLastIndexed(sheet);
    await saveFactSheet(avatarUrl, stamped);
    return stamped;
}


// =============================================================================
// SECTION 7: Pre-Merge Backup Snapshots
// =============================================================================
//
// Before each pipeline run mutates a fact sheet, we save a timestamped copy
// of the pre-merge state to the user files directory. This gives us:
//
//   - A safety net if the compactor ever does something we don't like (the
//     previous saved sheet is one file-rename away from restoration).
//   - A historical timeline we can analyse later — diff backup N against
//     backup N+1 to see exactly which facts were added, merged, or dropped
//     across the run. Studying how the sheet evolves over weeks is one of
//     the long-term goals of the project.
//
// Filename convention:
//   memPipeline_backup_<safeCharName>_<YYYY-MM-DD_HHMM>.json
//
// Date AND time are included so multiple runs in a single day (e.g. the
// scheduled 03:00 run plus a manual "Run Now" later) don't overwrite each
// other. Lexicographic sort on the filename also produces chronological
// order, which is convenient for any future scan tooling.
//
// Pruning (removing old backups when over a retention limit) is intentionally
// not implemented yet: ST's user files API exposes upload + verify + GET
// but its delete endpoint isn't clearly documented for user files
// specifically, and the documented endpoints with "delete" in the name
// (/api/chats/delete, /api/extensions/delete) target unrelated systems.
// Until that's confirmed, the retention setting exists in the UI as a knob
// for the future, and `pruneOldBackups` below is a no-op stub with a TODO.
// Default retention is 0 (unlimited), so this no-op is currently the user's
// chosen behaviour anyway.
// =============================================================================

/**
 * Derives the filename for a fact-sheet backup taken at a given moment.
 * Uses the same safe-character sanitisation as the live sheet, then appends
 * a YYYY-MM-DD_HHMM timestamp.
 *
 * @param {string} avatarUrl  - Character avatar filename, e.g. "Fin.png"
 * @param {Date}   timestamp  - When the backup is being taken (default: now)
 * @returns {string}          - e.g. "memPipeline_backup_Fin_2026-05-21_0300.json"
 */
function deriveBackupFileName(avatarUrl, timestamp = new Date()) {
    const baseName = avatarUrl.replace(/\.[^/.]+$/, '');
    const safeName = baseName.replace(/[^a-zA-Z0-9_\-.]/g, '_');

    const pad  = n => String(n).padStart(2, '0');
    const yyyy = timestamp.getFullYear();
    const mm   = pad(timestamp.getMonth() + 1);
    const dd   = pad(timestamp.getDate());
    const hh   = pad(timestamp.getHours());
    const mi   = pad(timestamp.getMinutes());

    return `${BACKUP_PREFIX}${safeName}_${yyyy}-${mm}-${dd}_${hh}${mi}.json`;
}

/**
 * Saves a snapshot of the given fact sheet to the user files directory as
 * a timestamped backup. Intended to be called BEFORE the pipeline mutates
 * the sheet (i.e. before mergeFactSheet runs), so the snapshot captures
 * the previous saved state exactly.
 *
 * IMPORTANT: pass a deep-clone of the sheet, not the same reference the
 * pipeline will mutate. JSON.parse(JSON.stringify(sheet)) is enough.
 *
 * This function NEVER throws. A failed backup logs a warning and returns
 * null — the pipeline must not be blocked by a backup failure.
 *
 * @param {string}      avatarUrl  - Character avatar filename
 * @param {object|null} sheet      - The pre-merge sheet to snapshot
 * @returns {Promise<string|null>} - Path of the saved file, or null on failure
 */
async function backupFactSheet(avatarUrl, sheet) {
    if (!sheet) return null; // no existing sheet → nothing to back up (first run)

    const fileName = deriveBackupFileName(avatarUrl);

    const json   = JSON.stringify(sheet, null, 2);
    const base64 = btoa(unescape(encodeURIComponent(json)));

    try {
        const response = await fetch('/api/files/upload', {
            method: 'POST',
            headers: getRequestHeaders(),
            body: JSON.stringify({
                name: fileName,
                data: base64,
            }),
        });

        if (!response.ok) {
            const msg = await response.text();
            console.warn(`${LOG_PREFIX} Backup failed for "${avatarUrl}": HTTP ${response.status}: ${msg}`);
            return null;
        }

        const result = await response.json();
        console.log(`${LOG_PREFIX} Backed up fact sheet for "${avatarUrl}" → ${result.path}`);
        return result.path;
    } catch (err) {
        // Network errors, unparseable response, etc. — log but don't propagate.
        console.warn(`${LOG_PREFIX} Backup threw for "${avatarUrl}": ${err.message}`);
        return null;
    }
}

/**
 * Stub for retention enforcement — currently a no-op.
 *
 * Once we confirm the right ST endpoint for deleting a user file, this will:
 *   1. List backups matching `memPipeline_backup_<safeCharName>_*` (no list
 *      endpoint is documented; we may need to maintain our own index, or
 *      probe likely files via /api/files/verify with a date-range guess).
 *   2. Sort by the embedded timestamp in the filename (lex order works).
 *   3. Delete the oldest ones until at most `keep` remain. retentionCount
 *      of 0 means unlimited — never prune.
 *
 * Callers can wire this in already; it just won't do anything until the
 * implementation lands.
 *
 * @param {string} avatarUrl
 * @param {number} keep        - max backups to retain; 0 = unlimited
 * @returns {Promise<number>}  - count of backups pruned (currently always 0)
 */
async function pruneOldBackups(avatarUrl, keep = 0) {
    if (!keep || keep <= 0) return 0; // unlimited retention — never prune
    // TODO(v2.x): Implement once ST's user-file delete endpoint is verified.
    // For now this is a no-op so the pipeline still completes cleanly when
    // a non-zero retention is configured. Old backups will accumulate until
    // pruning is wired in (or the user cleans them manually from disk).
    console.log(`${LOG_PREFIX} pruneOldBackups: retention=${keep} requested, but pruning is not yet implemented (see SECTION 7 notes). Skipping.`);
    return 0;
}


// =============================================================================
// Exports
// =============================================================================

export {
    deriveFileName,
    loadFactSheet,
    saveFactSheet,
    getLastIndexed,
    stampLastIndexed,
    formatFactSheetForContext,
    finaliseAndSave,
    deriveBackupFileName,
    backupFactSheet,
    pruneOldBackups,
};
