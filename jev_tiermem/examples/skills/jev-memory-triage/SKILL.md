---
name: jev-memory-triage
description: Diagnose missing or insufficient recall in Jev TierMem using the session's stored notes, raw records, and retrieval trace. Use when an agent reports that earlier messages or tool results cannot be found.
---

# Jev TierMem recall triage

Locate the TierMem checkout and the exact `--store` and `--session` used by the failing agent. A workspace path permits file reads; it does not import that directory into memory. Smoke runs have a separate store and the session `smoke`.

From the checkout, inspect local history without making provider calls:

```bash
jev_tiermem/.venv/bin/python -m jev_tiermem \
  --store "$MEMORY_STORE" --session "$MEMORY_SESSION" search "$SEARCH_TERMS"

jev_tiermem/.venv/bin/python -m jev_tiermem \
  --store "$MEMORY_STORE" --session "$MEMORY_SESSION" show
```

Set those three variables from the user's actual run and a distinctive file name, test name, or phrase. Check whether a matching original `tool` or `user` record exists. A record containing only the current question or a previous unsuccessful recall response is not evidence of the missing event.

If the original record is absent, check whether ingestion or `read_file` happened, whether the store/session changed, and whether the expected record belongs to a separate smoke run. Report which condition the observed data supports; do not guess.

If raw records exist, inspect `agent_trace.jsonl` in the same session directory. Compare the recalled raw IDs, summary IDs, `sufficient`, decision errors, and the search stop reason. A provider error, an insufficient summary, an exhausted search budget, and a lexical miss require different fixes. Do not treat a low sufficiency score alone as a network error.

The agent's `status: complete` means that its turn ended; it does not prove successful retrieval. Cite the original evidence IDs supporting the diagnosis and state any remaining gap. Preserve stored history during diagnosis. Ingestion or changing a threshold is a separate action, not proof that earlier retrieval succeeded.
