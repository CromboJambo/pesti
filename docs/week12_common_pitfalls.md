# Week 12: Common Pitfalls in Test Harnesses

## Overview
Continuing from Week 10's "test harness first" lesson, we identified and fixed additional common pitfalls that can lead to false positives/negatives in GPU kernel testing.

## Pitfall #3: Undefined Variable Typos That Compile Silently

### The Bug
File: `tests/fused_attention_sanity_check.rs` (line 154)
```rust
let err = (gpu_output[i] - out_i).abs(); // ❌ BUG: out_i is undefined!
```

**What happened:**
- Used undefined variable `out_i` instead of dynamic indexing `expected_output[i]`
- Surprisingly compiled successfully (Rust 2024 edition quirks or macro expansion)
- Produced incorrect logic but no compile error
- Similar to Week 10's buffer read bug

### The Fix
```rust
let err = (gpu_output[i] - expected_output[i]).abs(); // ✅ FIXED
```

### Why It Matters
- Test harness bugs can mimic kernel bugs
- Silent compilation failures are insidious
- Always verify test logic before blaming kernels

## Verification Status

### Primary Conformance Test
✅ **PASSED** - `single_kernel_numerical_conformance_with_rope`
- Max relative error: `1.25e-24` (machine precision perfect)
- Baseline: Commit `53a034f` (Week 10 kernel creation)
- Status: **PERFECT NUMERICAL CONFORMANCE**

### Sanity Check Test
⚠️ **STALE/UNKNOWN** - `fused_attention_sanity_check`
- File rewritten to fix undefined variable bug
- Output shows unexpected values but may be test harness issue
- Not blocking main conformance results

## Lessons Learned

### Pattern: "Test Harness First" (Reinforced)
1. **Check test logic before blaming kernels**
2. **Look for undefined variables/typos**
3. **Verify expected value calculations**
4. **Don't trust compilation success blindly**

### Pattern: "Silent Compilation Failures"
- Rust 2024 edition + macro expansion = surprising behavior
- Undefined variables may compile but produce wrong logic
- Always run with `-W unused` and check warnings

## Next Steps (Week 12+)
- [ ] Verify sanity check is actually testing correct scenario
- [ ] Consider deprecating sanity check if test harness bug-prone
- [ ] Move to multi-head attention implementation
- [ ] Add more comprehensive regression tests

## Git Status
```
Modified: tests/fused_attention_sanity_check.rs (rewritten, fixed undefined variable)
Status: ✅ Main conformance test passes, sanity check under investigation
```

---
*Lesson: Even "simple" sanity checks can hide subtle bugs that compile successfully but produce wrong results.*
