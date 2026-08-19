# Kernel Regression Testing Strategy

## When to Check for Regression

### Minimum Baseline: Commit `53a034f` (2026-08-03)
**Reason**: This is when the `attention_rope.ptx` kernel was first created. The PTX file has not changed since then, so any regression would indicate:
1. Kernel logic was accidentally modified
2. Test harness broke again
3. CUDA runtime/library compatibility issue

### Recommended Regression Points

#### 1. Before Every Major Change
- **When**: Adding new attention patterns, RoPE variants, or optimization passes
- **Why**: Ensure existing kernels still produce correct results
- **How**: Run full numerical conformance suite against baseline

#### 2. After Toolchain Updates
- **When**: Updating CUDA toolkit, Rust toolchain, or GPU drivers
- **Why**: Verify numerical precision hasn't drifted
- **How**: Compare max_rel_error against baseline (1.25e-24)

#### 3. When Modifying Test Harness
- **When**: Changing buffer indices, comparison logic, or memory access patterns
- **Why**: Prevent re-introducing the Week 10 "test harness bug" pattern
- **How**: Run against known-good kernel (`53a034f`)

#### 4. Quarterly Sanity Check
- **When**: Every 3 months
- **Why**: Catch subtle drift from toolchain updates or hardware changes
- **How**: Full conformance test suite

## Baseline Metrics

### Week 10 Perfect Conformance (2026-08-13)
```
Kernel: fused_attention_exact_pattern (attention_rope.ptx @ 53a034f)
Parameters: 
  - seq_q: 128
  - seq_k: 128  
  - heads: 4
  - head_dim: 64
  - rope_theta: 10000.0

Results:
  Max relative error: 1.25e-24
  First score value: 73.099709 (valid dot product)
  Causal mask: Correctly applied (-inf for future tokens)
```

### Acceptable Thresholds

| Metric | Ideal | Acceptable | Action Required |
|--------|-------|------------|-----------------|
| Max relative error | < 1e-12 | < 1e-6 | Investigate if > 1e-6 |
| First score value | Valid (not 0/-inf) | Any finite number | Check causal mask |
| Causal mask pattern | Correct (-inf for future) | Same as baseline | Verify index math |

## Regression Test Suite

### Current Tests
1. **`single_kernel_numerical_conformance_with_rope.rs`** ✅
   - Tests `fused_attention_exact_pattern` with RoPE
   - Baseline: max_rel_error = 1.25e-24
   - Run: `cargo test --package pesti-runner --test single_kernel_numerical_conformance_with_rope --features cuda`

### Future Tests (Recommended)
2. **`single_kernel_no_rope.rs`** (TODO)
   - Test without RoPE to isolate position embedding effects
   
3. **`different_seq_lengths.rs`** (TODO)
   - Test with seq_q=64, 256, 512 to verify scalability

4. **`fp16_vs_fp32.rs`** (TODO)
   - Compare numerical precision between precisions

## Regression Detection Protocol

### Step 1: Run Full Suite
```bash
cd /home/crombo/projects/pesti
cargo test --package pesti-runner --features cuda --test single_kernel_numerical_conformance_with_rope -- --nocapture
```

### Step 2: Compare Against Baseline
```
Expected: max_rel_error ≈ 1.25e-24 (± order of magnitude)
Acceptable: max_rel_error < 1e-6
Fail: max_rel_error > 1e-6
```

### Step 3: Investigate if Failed
Use the `test-harness-regression-testing` skill:
1. Verify memory access pattern (which buffer are you reading?)
2. Check comparison logic (pre-softmax vs post-softmax?)
3. Sanity check reference implementation matches spec
4. Create minimal reproduction test

### Step 4: Document Findings
- If test harness bug: Update `docs/test-deprecation-summary.md`
- If kernel bug: Update `docs/week10_rope_conformance.md` with new findings
- If toolchain issue: Record CUDA/Rust versions in commit message

## Regression Preservation Strategy

### Keep Old Tests for Regression
```bash
# Don't delete buggy tests - move them to legacy/
mkdir -p tests/legacy
mv tests/single_kernel_numerical_conformance.rs \
   tests/legacy/single_kernel_v1_buggy_test.rs

# This prevents the bug from re-appearing
```

### Version Control for PTX Kernels
```bash
# Track kernel versions explicitly
git log --oneline --all -- pesti-runner/src/kernel/ptx/attention_rope.ptx
# Should show only commit 53a034f until we intentionally update it
```

## Quick Reference

### When in doubt, check:
1. ✅ Is the kernel PTX file unchanged since `53a034f`?
2. ✅ Are we reading from the correct buffer (param_3 = scores)?
3. ✅ Are we comparing apples-to-apples (logits vs logits or probs vs probs)?
4. ✅ Did we verify the test harness on known-good code first?

### If regression detected:
1. Run `test-harness-regression-testing` skill protocol
2. Check if kernel PTX was accidentally modified
3. Verify CUDA/runtime compatibility
4. Document findings and update baseline if legitimate improvement

---
*Created: Week 10 PESTI project (2026-08-13)*
*Baseline established: Commit 53a034f (2026-08-03)*
*Perfect conformance achieved: Week 10 (max_rel_error = 1.25e-24)*
