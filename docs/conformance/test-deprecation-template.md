# Test Deprecation Template

Use this template when marking tests as deprecated after discovering test harness bugs.

## Example: Week 10 Test Fix

### Original Buggy Test (Week 9)
```rust
// BUGGY VERSION - reads from wrong buffer
let scores_host_f32: Vec<f32> = scores_host.iter()
    .map(|&x| x as f32)
    .collect();

// BUGGY VERSION - compares logits to probabilities
for (i, &score_val) in scores_host_f32.iter().enumerate() {
    let expected = llama_probs[i] as f32;
    // Comparing pre-softmax logits to post-softmax probs!
}
```

### Fixed Version (Week 10)
```rust
// FIXED VERSION - reads from correct buffer (param_3 = scores)
let scores_host_f32: Vec<f32> = scores_host.iter()
    .map(|&x| x as f32)
    .collect();

// FIXED VERSION - applies softmax before comparison
for (i, &score_val) in scores_host_f32.iter().enumerate() {
    let expected = llama_probs[i] as f32;
    // Apply softmax to kernel logits first!
}
```

## Deprecation Comment Format

```rust
// [DEPRECATED 2026-08-13] 
// Bug: Reading from output buffer (param_5) instead of scores buffer (param_3)
// Symptom: Kernel appeared to output zeros/-inf values
// Root cause: Test harness bug, not kernel bug
// Fix: Read from &*scores_host as f32, apply softmax before comparison
// See: docs/week10_rope_conformance.md, commit 04f9b7d
```

## Migration Path

1. **Document the bug** in the test file comment
2. **Mark as deprecated** with date and commit reference
3. **Provide corrected logic** in new test file
4. **Keep old test** for regression (prevent re-introduction)
5. **Update documentation** explaining the fix

## Regression Preservation

To prevent the bug from re-appearing:

```bash
# Keep buggy version in separate file
mv tests/single_kernel_numerical_conformance.rs \
   tests/legacy/single_kernel_numerical_conformance_v1_buggy.rs

# Create new corrected version
cp tests/single_kernel_numerical_conformance_v1_buggy.rs \
   tests/single_kernel_numerical_conformance_with_rope.rs
```

## Checklist for Deprecation

- [ ] Document the bug with specific symptoms
- [ ] Add date and commit reference to deprecation comment
- [ ] Provide corrected comparison logic
- [ ] Keep old test for regression (in legacy/ folder)
- [ ] Update documentation (docs/week10_*.md)
- [ ] Verify new test passes on known-good kernel
- [ ] Commit with clear message explaining the fix

---
*Template created: Week 10 PESTI project (2026-08-13)*
