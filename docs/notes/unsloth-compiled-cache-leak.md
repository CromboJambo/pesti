# Unsloth Compiled Cache: A Peek Under the Hood

**Date**: 2026-08-30
**Trigger**: `unsloth_compiled_cache/moe_utils.py` (79KB) appeared untracked in the
crabjar repo. Not a security leak — a CWD-relative cache write from the inference
server. Documented here because it exposes how Unsloth's MoE path works internally.

## What the artifact is

`unsloth_compiled_cache/moe_utils.py` is a **torch-compiled copy of Unsloth's own
`moe_utils.py`** — the expert-routing utilities for MoE models. It is not model
weights, not user data, not a secret. It is a build artifact: Unsloth pre-compiles
its MoE kernels (via `torch.compile`) and drops the compiled module into a cache
directory so subsequent loads skip recompilation.

Header of the leaked file:

```python
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team.
# ...
UNSLOTH_COMPILE_LOCATION = os.environ.get(
    "UNSLOTH_COMPILE_LOCATION", "unsloth_compiled_cache"
)
```

That last line is the whole story.

## The mechanism

1. The env var `UNSLOTH_COMPILE_LOCATION` defaults to the **relative** string
   `"unsloth_compiled_cache"`.
2. Python resolves relative paths against the **process CWD**.
3. The `unsloth` studio launcher (`unsloth start hermes --model
   unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M`) was started from inside
   `/home/crombo/projects/crabjar`.
4. At first model load (server start), the MoE compile step wrote the cache into
   the CWD → `crabjar/unsloth_compiled_cache/moe_utils.py`.

Verification (all observed live):

- Cache file mtime: `Aug 30 10:34` — matches the `unsloth run` server process
  start time to the minute.
- `readlink /proc/<launcher-pid>/cwd` → `/home/crombo/projects/crabjar`.
- The serving chain at the time: `unsloth start` (launcher) → `unsloth run -H
  localhost -p 8888` (API shim) → `llama-server -m Qwen3.8-27B-UD-Q4_K_M.gguf
  --port 34427 --parallel 4 --spec-type draft-mtp` (actual inference).
- The model is Qwen3.8-27B, a MoE model — which is exactly why the MoE compile
  path fired. A dense model would never touch this code path.

## Why it's the same class of bug as node_modules

Any tool that writes build artifacts to a CWD-relative default path will leak
into whatever repo it was launched from. `node_modules/`, `__pycache__/`,
`.pytest_cache/` — all the same pattern. The fix is always the same two layers:

1. **Repo side** (defense): gitignore the directory. pesti already has
   `unsloth_compiled_cache/` in `.gitignore` (line 18) — crabjar needed the same.
2. **Root cause** (proper fix): set `UNSLOTH_COMPILE_LOCATION` to an absolute
   path, e.g. `~/.cache/unsloth/compiled`, in the server's environment. Then the
   cache lands in the cache dir regardless of launch directory.

## Notes for pesti

- pesti hits this too: it has its own `unsloth_compiled_cache/` (already
  gitignored). Any workflow that starts Unsloth from inside a checkout will
  regenerate it.
- If pesti ever wants a torch.compile-style kernel cache for its own CPU/GPU
  kernels, the lesson is: **never default a cache path to a relative string**.
  Default to `$XDG_CACHE_HOME` or `~/.cache/<tool>/`, and only honor a relative
  override when it's explicitly set.
- The compiled `moe_utils.py` itself is a useful reference: it shows Unsloth's
  MoE expert-routing implementation (gating, top-k selection, expert dispatch)
  in plain Python, which is a decent oracle when writing equivalent Rust in
  pesti-runner.
