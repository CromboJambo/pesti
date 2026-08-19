# Quick Reference: Kernel Regression Testing

## One-Liner Check (5 seconds)
```bash
cd /home/crombo/projects/pesti && \
cargo test --package pesti-runner --test single_kernel_numerical_conformance_with_rope --features cuda -- --nocapture 2>&1 | grep -E "Max relative error|test result"
```

## Expected Output
```
Max relative error: 1.25e-24
test result: ok. 1 passed; 0 failed
```

## Acceptable Thresholds

| Metric | Ideal | Acceptable | Action if Failed |
|--------|-------|------------|------------------|
| Max rel error | ≈1.25e-24 | <1e-6 | Run `test-harness-regression-testing` skill |
| First score | 73.099709 | Any finite number | Check causal mask |
| Test result | ok | ok | Investigate immediately |

## If Regression Detected

### Step 1: Verify Kernel PTX Unchanged
```bash
git log --oneline --all -- pesti-runner/src/kernel/ptx/attention_rope.ptx
# Should show ONLY: 53a034f Add specialized PTX kernels...
```

### Step 2: Run Full Debugging Protocol
```bash
skill_view(name='test-harness-regression-testing')
# Follow the 4-step verification process
```

### Step 3: Check Common Pitfalls
- [ ] Reading from correct buffer (param_3 = scores, not param_5 = output)
- [ ] Comparing apples-to-apples (logits vs logits, not logits vs probs)
- [ ] Using relative error for floating-point stability
- [ ] Verified test harness on known-good code first

### Step 4: Document Findings
Update `docs/test-deprecation-summary.md` with new findings

## When to Run

- ✅ Before adding new attention patterns/RoPE variants
- ✅ After CUDA/Rust/toolchain updates
- ✅ When modifying test harness code
- ✅ Quarterly sanity check (every 3 months)

## Baseline Info

**Kernel**: `attention_rope.ptx @ commit 53a034f`  
**Created**: August 3, 2026  
**Tests passed**: Week 10 perfect conformance (max_rel_error = 1.25e-24)

## Documentation Links

- Full strategy: `docs/kernel-regression-strategy.md`
- Deprecation template: `docs/test-deprecation-template.md`
- Summary document: `docs/regression-testing-summary.md`
- Week 10 case study: `docs/week10_rope_conformance.md`

---
*Created: Week 10 PESTI project (2026-08-13)*
