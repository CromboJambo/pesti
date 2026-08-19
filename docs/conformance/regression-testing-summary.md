# Regression Testing Summary - Week 10

## How Far Back to Check?

**Answer**: From commit **`53a034f`** (August 3, 2026) forward

### Why This Commit?

This is when the `attention_rope.ptx` kernel was first created:
```bash
$ git log -1 --format="%H %s %ai" 53a034f
53a034f5c0602e79677d784d3c97926326de3b6c Add specialized PTX kernels for attention acceleration 2026-08-03 15:26:49 -0500
```

**Key insight**: The kernel PTX file has **never changed** since then (only 1 commit in git history). This means:
- ✅ The kernel logic is frozen and stable
- ✅ Any regression would mean we accidentally modified the PTX or broke our test harness
- ✅ We only need to check against this baseline, not every historical commit

## Baseline Metrics (Week 10 Perfect Conformance)

```
Kernel: fused_attention_exact_pattern @ 53a034f
Parameters: seq_q=2, seq_k=32, heads=4, dim=16
Results:
  Max relative error: 1.25e-24 ✅ (PERFECT!)
  First score value: 73.099709 (valid dot product)
  Causal mask: Correctly applied (-inf for future tokens)
```

## Regression Test Script

**Location**: `/tmp/hermes-run-regression-test.sh` (copy to project if needed)

**Usage**:
```bash
cd /home/crombo/projects/pesti
/tmp/hermes-run-regression-test.sh
```

**Expected output**:
```
=== Kernel Regression Test ===
Baseline: Commit 53a034f (attention_rope.ptx)
Expected max_rel_error: ~1.25e-24 (acceptable < 1e-6)

✅ REGRESSION TEST PASSED
```

## When to Run Regression Tests

### Minimum Frequency
1. **Before every major change** (new attention patterns, RoPE variants)
2. **After toolchain updates** (CUDA, Rust, GPU drivers)
3. **When modifying test harness** (buffer indices, comparison logic)
4. **Quarterly sanity check** (catch subtle drift)

### Quick Check (5 seconds)
```bash
cargo test --package pesti-runner \
    --test single_kernel_numerical_conformance_with_rope \
    --features cuda \
    -- --nocapture 2>&1 | grep -E "Max relative error|test result"
```

**Acceptable threshold**: `max_rel_error < 1e-6`  
**Ideal threshold**: `max_rel_error ≈ 1.25e-24` (same as baseline)

## Documentation Created

### 1. Regression Strategy Document
**Location**: `docs/kernel-regression-strategy.md`
- When to check for regression
- Baseline metrics and acceptable thresholds
- Step-by-step protocol if regression detected
- Preservation strategy for old tests

### 2. Test Deprecation Template
**Location**: `docs/test-deprecation-template.md`
- Format for marking buggy tests as deprecated
- Migration path for corrected comparison logic
- Regression preservation checklist

### 3. Summary Document
**Location**: `docs/test-deprecation-summary.md` (force-committed)
- Week 10 case study (test harness bug vs kernel bug)
- "Test harness first" debugging principle
- When to trust your test vs when to debug it

### 4. Regression Test Script
**Location**: `/tmp/hermes-run-regression-test.sh`
- Automated comparison against baseline
- Extracts key metrics from test output
- Returns success/failure status

## Key Takeaway

**You don't need to check every historical commit!**

Since the kernel PTX is frozen at `53a034f`, you only need to:
1. ✅ Run the conformance test against this baseline
2. ✅ Compare max_rel_error against 1.25e-24 (± order of magnitude)
3. ✅ If it passes, your kernel is still correct!

**The real risk**: Not that the kernel changed, but that **your test harness broke again** (like Week 10!). That's why we created the `test-harness-regression-testing` skill to catch those bugs early.

---
*Created: Week 10 PESTI project (2026-08-13)*
*Baseline established: Commit 53a034f (August 3, 2026)*
*Perfect conformance achieved: Week 10 (max_rel_error = 1.25e-24)*
