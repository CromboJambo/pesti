# Release Checklist for pesti-gguf

## Pre-Publish Steps ✅

- [x] All 49 tests passing
- [x] Real-file conformance tests verified (Qwen2.5 0.5B & 3B)
- [x] Performance benchmarks documented (36.7ms for 0.5B, 33.4ms for 3B)
- [x] README.md with marketing copy and performance claims
- [x] PERFORMANCE.md with detailed benchmark methodology
- [x] CONFORMANCE.md with test coverage details
- [x] Benchmarks added (cargo bench infrastructure)
- [x] BufReader optimization applied for fast I/O

## To Do Before Publishing

### 1. Update Version Number
```bash
# In Cargo.toml, change:
version = "0.1.0" → version = "0.2.0"
```

### 2. Check Documentation
```bash
# Build docs locally
cargo doc --open

# Verify API documentation renders correctly
```

### 3. Test Publish Dry Run
```bash
# Dry run to check for issues
cargo publish --dry-run

# Should show:
# "Uploading pesti-gguf v0.2.0 to registry"
# "Packaging done"
```

### 4. Final Checks
- [ ] Verify license file exists (LICENSE)
- [ ] Check `publish = true` in Cargo.toml
- [ ] Ensure repository URL is correct
- [ ] Run `cargo clippy` for final lint check

### 5. Publish to crates.io
```bash
# Login (first time only)
cargo login

# Publish
cargo publish

# Wait ~2 minutes for index update
```

### 6. Post-Publish Verification
```bash
# Check it's on crates.io
curl https://crates.io/api/v1/crates/pesti-gguf | jq '.crate.max_version'

# Test installation
cargo add pesti-gguf --dry-run
```

## Marketing Assets to Prepare

### Social Media Posts
- **Twitter/X**: "🎉 Just released `pesti-gguf` - a memory-safe, zero-dependency GGUF parser for Rust! 
  - ✅ 2x faster than llama.cpp FFI
  - ✅ 5x faster than Python gguf
  - ✅ Full v1/v2/v3 support
  - ✅ WASM-ready
  
  crates.io/crates/pesti-gguf"

- **Rust Forum**: "Hey r/rust! I've been building a GGUF parser for Rust developers who want to inspect model metadata without linking against llama.cpp. 

Key features:
- Pure Rust, no C++ dependencies
- Structured error handling (no more panic on invalid magic!)
- 36ms parse time for 0.5B models
- Full conformance with Qwen2.5

Would love feedback from the community!"

### GitHub README Updates
- Add badges for:
  - Crates.io version
  - Docs.rs documentation
  - License
  - Test status

## Long-Term Goals

### Phase 1 (Current)
- ✅ Standalone release with performance claims
- Build credibility with real users

### Phase 2 (3-6 months)
- Measure adoption (stars, downloads, issues)
- If >10 stars/month → continue independent development
- If someone asks for specific feature → validate demand

### Phase 3 (6+ months)
- **Decision point**: 
  - If Rust-native use cases are strong → stay independent
  - If llama.cpp maintainers ask for integration → contribute upstream
  - If adoption is flat → pivot to niche (e.g., GGUF validation service)

## Key Metrics to Track

1. **Adoption**
   - crates.io downloads/month
   - GitHub stars
   - Number of dependent crates

2. **Community Feedback**
   - Issues filed (quality indicator)
   - Feature requests (demand signal)
   - PRs from contributors (engagement)

3. **Performance**
   - Benchmark regression checks
   - WASM compilation size
   - Memory usage on large models

## Success Criteria

✅ **Good**: 50+ stars, 100+ downloads/month after 3 months  
✅ **Better**: 200+ stars, 500+ downloads/month, active contributors  
❌ **Bad**: <20 stars, <50 downloads/month after 6 months (pivot needed)

---

**Ready to publish?** Run: `cargo publish` 🚀
