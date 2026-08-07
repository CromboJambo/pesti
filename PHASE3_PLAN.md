# Phase 3: Upstream Contribution Plan

## Status: Ready to Execute

### Research Summary

After analyzing llama.cpp issues and PRs, I've identified three concrete opportunities where PESTI's learnings can help the ecosystem:

---

## Opportunity 1: Documentation - "GGUF Conformance Guide"

**Problem**: Users frequently encounter parsing errors with non-standard GGUF files (deepseek, nanbeige, etc.) due to architecture-specific tensor naming conventions.

**PESTI Learning**: 
- Built architecture-aware metadata extraction with fallback keys
- Implemented byte-exact dequantization verification
- Created comprehensive conformance tests

**Contribution**: 
Create `docs/gguf-conformance.md` that explains:
1. How GGUF metadata varies by model architecture
2. Common parsing error patterns and their causes
3. Verification steps (like PESTI's conformance testing)
4. Fallback strategies for non-standard models

**Effort**: ~2-4 hours
**Impact**: High - helps many users debug issues

---

## Opportunity 2: Tooling - Standalone GGUF Checker

**Problem**: Users need to verify GGUF files before loading them, but llama.cpp's errors are often cryptic.

**PESTI Learning**: 
- `pesti-conformance` library with detailed error reporting
- Byte-exact verification against reference implementations
- Architecture-specific test cases

**Contribution**:
Create a standalone CLI tool (Rust-based) that:
```bash
gguf-checker --path model.gguf --verbose
# Output:
# ✅ GGUF v3 valid
# ✅ Architecture: qwen2 (detected from metadata)
# ✅ Quantization: Q4_K_M (verified dequant)
# ⚠️  Warning: Non-standard tensor naming detected
# ℹ️  Fallback keys used: [llama.attention.layer_norm_epsilon]
```

**Effort**: ~8-12 hours
**Impact**: Medium-High - practical tool for users

---

## Opportunity 3: Parser Bug Fix (#24807)

**Problem**: llama.cpp silently drops valid tool calls when encountering malformed XML in parser.

**PESTI Learning**: 
- Built graceful fallback mechanisms (CPU ↔ GPU routing)
- Understands how parsers can fail at different stages

**Contribution**:
Modify `src/llama.cpp`'s grammar parser to:
1. Log warnings instead of silent drops
2. Attempt recovery from malformed XML
3. Add `--verbose-parser` flag for debugging

**Effort**: ~6-8 hours
**Impact**: Medium - specific but important UX improvement

---

## Recommended Execution Order

### Phase 3a: Documentation (Week 1)
1. Read existing llama.cpp docs (`docs/models.md`, `docs/function-calling.md`)
2. Draft "GGUF Conformance Guide" based on PESTI experience
3. Submit PR to llama.cpp/docs
4. **Goal**: Establish contribution credibility

### Phase 3b: Tooling (Week 2-3)
1. Create `gguf-checker` as separate repo (not in llama.cpp)
2. Reuse logic from `pesti-conformance` where applicable
3. Publish to crates.io and document usage
4. **Goal**: Build reputation as "GGUF expert"

### Phase 3c: Bug Fix (Week 4+)
1. Tackle #24807 once comfortable with llama.cpp codebase
2. Start with minimal fix (logging improvements)
3. Iterate based on maintainer feedback
4. **Goal**: Establish as reliable contributor

---

## Success Metrics

- [ ] First PR merged to llama.cpp/docs
- [ ] gguf-checker tool published and used by others
- [ ] Understanding llama.cpp internals well enough to contribute bug fixes
- [ ] Reputation as "the person who understands GGUF" (as per roadmap)

---

## What This Is NOT

- ❌ Trying to beat llama.cpp at benchmarks
- ❌ Forking llama.cpp for a competing product
- ❌ Becoming a maintainer immediately

## What This IS

- ✅ Applying PESTI learnings to help the ecosystem
- ✅ Building reputation through documentation and tooling
- ✅ Learning llama.cpp internals through contributions
- ✅ Preparing for deeper upstream work if desired

---

*Last updated: August 2026*  
*Based on Phase 2 completion (GPU integration via GEMM proxy)*
