# Memory Lattice

A complete long-term memory system for [SillyTavern](https://github.com/SillyTavern/SillyTavern), built for locally-hosted AI companions. One extension, both halves of memory:

- **The write path** runs nightly (or on demand): it reads your chat logs, writes first-person memories of each conversation, extracts durable facts into a versioned fact sheet, and indexes everything into a [Qdrant](https://qdrant.tech/) vector database.
- **The read path** runs on every message: your outgoing message is embedded, relevant memories are retrieved — each bringing its nearest-in-time neighbors so recall arrives with context attached — and injected into the prompt alongside the always-present fact sheet.

Chat logs remain the **canonical record**. Everything this extension produces — summaries, vectors, consolidated arcs — is derived data that can be rebuilt from the logs at any time. Nothing here ever modifies a chat log.

The design is documented in [a long-form architecture write-up](https://mlixer.github.io/spatiotemporal-memory-lattice/), and this extension is one component of [The Braid](https://github.com/mlixer/the-braid), a fully local companion framework.

## How memory is structured

| Layer | What it is | When it's made |
|---|---|---|
| **Fact sheet** | Durable facts in editable categories (people, world, preferences, projects, identity), each dated. Injected every message — identity is never a retrieval gamble. | Extracted per chunk, merged, periodically compacted by the LLM under a per-category cap |
| **Memories** | First-person, past-tense accounts of conversation chunks, refined with cross-chunk context, each carrying its real timestamp | Nightly pipeline / Run Now |
| **Consolidated arcs** | Topic threads that span many sessions, clustered by embedding similarity and merged into single narrative memories. A derived layer, rebuilt from scratch each run — originals untouched | After each index run |
| **Shelf** (optional) | Writing your companion's agent half produced offline (curiosities, gifts), ingested from a [ticket server](https://github.com/mlixer/ticket-server) by content-hash diff | Each pipeline pass |

## Requirements

- SillyTavern (recent release; the extension uses `generate_interceptor`)
- **Qdrant** reachable from the browser running ST (default `localhost:6333`)
- **An LLM endpoint** for summarization — any llama.cpp-style OpenAI-compatible `/v1/chat/completions` server (default port `8070`). This can be the same model that plays your character.
- **An embedding endpoint** — either:
  - **Ollama** (`/api/embeddings`, default port `11434`, e.g. `nomic-embed-text`), or
  - **any OpenAI-compatible server** (`/v1/embeddings`), e.g. llama.cpp serving a GGUF embedding model

All of it local. Nothing leaves your machine.

## Install

SillyTavern → **Extensions** → **Install extension** → paste this repository's URL. The manifest lives at the repo root, so the git install works directly.

Then open **Extensions → Memory Lattice** and check the connection settings against your setup.

## Quick start

1. Set your ports (summarization LLM, embedding server, Qdrant) and **Vector Size** to your embedding model's output dimensions (768 for `nomic-embed-text`).
2. Press **Test Qdrant** and **Test Embedding** — the embedding test reports the vector size it got back; it must match the Vector Size setting.
3. Open a character's chat and press **Run Now**. First run processes every session for that character — slow, many LLM calls. Watch the browser console for progress.
4. From then on the daily scheduler (default 03:00, tab must be open) runs incrementally — only new or changed sessions are reprocessed.
5. Chat. Recall happens automatically; tune it with **Preview Recall**.

## Tuning recall

Don't guess at thresholds — measure. The settings drawer has two instruments:

- **Preview Recall** runs the full read path for a test query and shows exactly what *would* be injected: each memory's similarity score, its age, and the token cost. Nothing is sent to the model.
- **Wide (top 20)** ignores your threshold and k entirely and shows the top 20 memories by raw similarity, flagging which fall below your threshold. When a memory you *know* is relevant isn't surfacing, this shows you where it actually ranks — so you can tell a threshold problem from an embedding-model problem before changing anything.

Settings worth knowing: **k** (memories per message), **Relevance Threshold**, **Memory Position** (injection depth from the end of chat), **Retain Recent** (fold prior messages into the query so short follow-ups retrieve well), and the **neighbor window** (how far in time each match may reach for its context).

## Changing embedding models

Vectors from different models don't mix — not just dimensions, the spaces themselves are incompatible. Migrating is a deliberate operation. The runbook, learned the hard way:

1. **Snapshot first**: `POST /collections/<name>/snapshots` on Qdrant for each `mem_*` collection. This is your revert path.
2. Back up the fact sheet (the pipeline does this automatically, but take your own copy of `memPipeline_<char>.json` from ST's user files on a migration day).
3. If the new model has **different dimensions**, delete each collection outright (`DELETE /collections/mem_<char>`) — the pipeline's wipe clears points but keeps the old schema, which would reject the new vectors.
4. Update **Embedding Model / URL / API / Vector Size** in the drawer, **Save Settings**, then **reload the tab** — a tab opened before the save keeps running the old settings. Close other ST tabs (including phones): any stale tab can answer the interceptor with old config.
5. **Test Embedding** in the fresh tab and confirm the reported dimension matches.
6. **Force Full Reindex**. The indexer has two guards for exactly this day: a 120-second timeout per embedding call (a wedged server once silently froze a 495-summary run), and a dimension tripwire that aborts loudly if vectors of the wrong size start coming back.

## Shelf ingest (optional)

If you run the [ticket server](https://github.com/mlixer/ticket-server) — the agent-execution half of The Braid — the Lattice can ingest your companion's own offline writing into the same memory collection. Entries are diffed by content hash, so re-runs are cheap and idempotent; files removed from the shelf are removed from memory. Off by default; enable it in the drawer once a ticket server is reachable.

## Prompts are yours (and your companion's)

Every prompt in the pipeline — memory writing, fact extraction, compaction, refinement, consolidation — is editable in the **Advanced** drawer, with per-prompt reset buttons and load-bearing tokens (`{{CATEGORIES}}`, `{{CAP}}`, …) that survive careless edits by being re-appended automatically. Fact categories are equally editable, with automatic sheet migration when keys are renamed.

The shipped prompts are deliberately plain. The intended path is that your companion eventually rewrites them in its own voice — memory, after all, should sound like the one remembering.

## Safety properties

- Chat logs are never written to — read-only canonical source.
- The fact sheet is backed up before every merge, every compaction, every manual edit, and every category migration. Timestamped copies accumulate in ST's user files.
- Every derived layer (vectors, consolidated arcs) can be rebuilt from scratch via **Force Full Reindex**.
- Recall is best-effort by doctrine: any failure in the read path degrades to no injection and one console warning. A memory outage never blocks a message.
- Incremental runs wipe and re-index only the sessions that changed; dormant sessions' memories are never touched.

## Version history

- **v3.1 — Memory Lattice.** Write and read paths merged into one extension. Per-message recall with temporal neighbor expansion, blended consolidated arcs, relative ages ("3w ago"), swipe-stable retrieval cache, preview/wide tuning instruments, fact-sheet editor, configurable embedding backend (Ollama or OpenAI-style), embed timeout + dimension tripwire, mid-run fact compaction for large reindexes.
- **v2.x — Memory Pipeline.** Write path: two-pass refinement, fact compaction under caps, editable prompts and categories, cross-session consolidation, gap-aware chunking, per-chunk timestamps, selective incremental wipes.
- **v1 — Memory Pipeline.** Summaries, fact sheet, nightly indexing.
