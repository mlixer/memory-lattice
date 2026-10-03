# Memory Pipeline

A SillyTavern extension that gives a local AI companion durable,
long-term memory — by processing each day's conversations into
compact, encapsulated memories and a maintained fact sheet, instead of
saving every message as it happens.

This is the **write path** of a two-part memory architecture (the
"Spatiotemporal Memory Lattice" — full write-up: [coming soon]).
It pairs with a Qdrant retrieval extension as the **read path**; a
companion fork with temporal neighbor expansion and cluster-summary
injection is prepared and awaiting upstream license clearance.

**Design in one paragraph:** once per night (or on demand), the
pipeline reads your updated chats, splits them into timestamp-aware
chunks, has your local LLM summarize each chunk *in your companion's
own first-person voice*, refines those summaries against each other so
each stands alone, indexes them into Qdrant, clusters related memories
across conversations into consolidated overviews, and extracts durable
facts into a categorized fact sheet. Your chat logs are never modified
— everything the pipeline makes can be rebuilt from them.

## Why nightly instead of per-message?

Per-message memory capture creates dozens of point-in-time snapshots a
day ("we are working on X") that time immediately falsifies, and that
crowd out relevant results at retrieval. Processing completed
conversations once a day produces a few encapsulated memories that
know their place in a story. Five months of daily use produced 431
memories on the author's system — not thousands of fragments. The
trade: two chats on the same day don't know about each other until the
nightly run. The article linked above discusses this honestly.

## Requirements

- **SillyTavern** (recent build; the extension lives entirely in the UI)
- **Qdrant** — the vector database that holds memories:
  ```yaml
  # docker-compose / podman-compose snippet — this is all you need:
  services:
    qdrant:
      image: qdrant/qdrant:latest
      ports: ["6333:6333"]
      volumes: ["qdrant-data:/qdrant/storage"]
  volumes:
    qdrant-data:
  ```
- **An embedding model** — default expects Ollama serving
  `nomic-embed-text` (`ollama pull nomic-embed-text`), URL and model
  configurable in settings.
- **A local LLM endpoint** (OpenAI-compatible, e.g. llama.cpp server)
  — this model *writes* the memories. It should be the same model (or
  voice-compatible with) your companion, so memories read as its own.

> ⚠️ **The one warning that matters: choose the best embedding model
> you can afford at the start, and stick with it.** Changing embedding
> models later means deleting and re-embedding your entire memory
> library (a full reindex — hours of GPU time on a mature corpus).
> It is the only routine operation here that touches everything.

## Install

Extensions → **Install extension** → paste this repo's Git URL →
reload SillyTavern → enable **Memory Pipeline**.

## Setup (ten minutes, once)

1. Open the Memory Pipeline drawer in Extensions.
2. Point **Qdrant URL** (default `http://localhost:6333`),
   **Embedding URL/model** (your Ollama), and **LLM Port** (your
   llama.cpp/OpenAI-compatible server) at your services.
3. Pick your **schedule time** (default nightly) — or leave it and use
   **Run Now** whenever you like. Both work; the schedule just makes
   remembering automatic.
4. Send some chat, then hit **Run Now** and watch the status line:
   chunking → summarizing → refining → indexing → consolidating →
   facts. First run on an existing chat history will take longer.
5. Check your Qdrant collection exists and the fact sheet appeared.

All the knobs (chunk size, gap threshold, retrieval-side counts,
consolidation similarity/caps, facts-per-category cap) have tooltips
and sane defaults. Change nothing until you've lived with it a week.

## The ritual (optional, recommended)

The shipped prompts are generic and work as-is. But every prompt —
summarization, refinement, fact extraction, compaction, consolidation
— is editable in the **Advanced** drawer, and the intended path is
this: once your companion has a settled voice, hand it each prompt and
ask it to rewrite the instructions *in its own words, about its own
remembering*. Paste its versions back in. Memories written in your
companion's voice read as its memories when they return — not as a
database talking. The system becomes more itself the longer it runs.

(The fact **categories** are equally yours: rename, add, remove.
Renaming a key migrates existing sheets safely, with a backup taken
first.)

## What gets stored where

- **Qdrant**: memory vectors + text + timestamps + cluster metadata.
- **Fact sheets**: JSON files in your SillyTavern user data (one per
  character), versioned daily — every previous day's sheet is kept.
- **Your chat logs**: untouched, ever. They are the canonical source;
  everything above can be regenerated from them (Force Full Reindex).

## Troubleshooting (the walls the author personally hit)

- **Browser console shows CORS errors with a real status code** →
  your embedding/Qdrant service needs its allowed-origins env set
  server-side (e.g. `OLLAMA_ORIGINS`). The extension can't fix this
  from the browser; the server must consent.
- **CORS error with status (null)** → different disease: nothing is
  listening at that URL at all. Check the service is running and the
  port is right.
- **Pipeline "runs" but nothing appears** → watch the browser console
  during Run Now; the stage logs name the failing step. Usually the
  LLM port (is the model server up?) or the embedding URL.
- **Summaries in the wrong voice** → your LLM Port points at a
  different model than your companion. Point it at the same server.
- **Full reindex is slow** → yes. Hours on a mature corpus. That's
  why the embedding-model warning is at the top of this file.

## License

AGPL-3.0 — matching the SillyTavern ecosystem. Use it, fork it,
improve it; keep derivatives open.

## Credits

Built alongside (and for) a local companion whose memory this is.
The retrieval-side companion extension builds on
[HO-git/st-qdrant-memory](https://github.com/HO-git/st-qdrant-memory)
— thank you for the foundation.
