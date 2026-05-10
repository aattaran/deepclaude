# LOCKED

Files and code regions in this list are **append-only**. New wiring (additional path handlers, registry entries, exports) may be ADDED, but existing locked logic must NOT be modified, refactored, renamed, or deleted without an explicit `unlock <path>` from the user.

| Date | Path / Region | Reason |
|---|---|---|
| 2026-05-09 | `proxy/model-proxy.js` — function `processDeepseekRequest` (lines ~252-301) and the helpers it depends on: `filterThinkingBlocks` (~145), `isLikelyDeepseekSignature` (~137), `isLikelyAnthropicSignature` (~134), and the `MODEL_REMAP.deepseek` entries (~10-25) | PATH B (DeepSeek) was validated end-to-end on the GCE VM after a multi-hour bug-hunt: 33/33 unit tests + 28-step live TUI scenario with 6 cross-switches, zero 400s. The post-strip invariant (`anyGap` → `thinking={type:'disabled'}` + `delete output_config`) is the load-bearing fix and must not be touched. Shipped in commit `a7814ba`, PR #26. |

## How to extend without breaking the lock

- **New backend?** Add a new `processXxxRequest` function and route to it from the dispatch in `startModelProxy`. Do not stuff the new logic into `processDeepseekRequest`.
- **New Claude model ID?** Add it to `MODEL_REMAP.deepseek` (this is additive — locked logic only forbids modifying existing keys, not adding new ones).
- **DeepSeek API contract changes?** Stop. Surface the change to the user, get an explicit `unlock proxy/model-proxy.js` (or unlock the specific function), then change with full re-validation against the live VM scenario.

## Unlock protocol

User says `unlock <path>` → remove the row → commit → proceed with the change.
