# Crabjar Requirements for pesti Integration

**Status:** Proposed  
**Author:** crabjar team  
**Date:** November 7, 2026  
**Target:** pesti structural tokenizer and semantic analyzers  

## Overview

Crabjar uses pesti's structural code tokenization and semantic analysis as the foundation for its JEV (Just Enough Verification) decision-making system. This document outlines crabjar's requirements and improvement requests for pesti to enable production-grade integration.

## Current Integration Status

- **Working:** `analyze_errors()`, `analyze_mutability()`, `analyze_ownership()`, `analyze_complexity()` in `crabjar-conductor/src/decision/structural.rs`
- **Examples:** `crabjar-conductor/examples/jev_pesti_integration.rs`, `security_pr_test.rs`
- **Criteria:** `specs/criteria/code-review-risk.json` uses pesti structural patterns

## Requirements

### 1. Binary Distribution (P0) - ✅ COMPLETE

**Status:** Published to crates.io as `pesti-structural-tokenizer` v0.1.4 (November 7, 2026)

**Completed:**
- [x] Publish to crates.io as `pesti-structural-tokenizer` with versioned releases
- [x] Document installation in pesti README

### 2. API Stability (P0) - IN PROGRESS

**Problem:** Semantic analyzer APIs are not formally versioned. Crabjar's JEV pipeline depends on consistent behavior across pesti updates.

**Request:**
- Formalize `pesti_structural_tokenizer::semantic` module as stable API
- Document breaking change policy
- Provide migration guide for major version changes

### 3. Missing Semantic Analyzers (P1) - ✅ COMPLETE

**Status:** Both analyzers implemented and tested

**Completed:**
- [x] `analyze_ownership()` — detects owned/borrowed/mutably_borrowed patterns via lightweight string scanning
- [x] `analyze_complexity()` — cyclomatic complexity, max nesting depth, control flow statement count (published in v0.1.4)

### 4. Performance at Scale (P1)

**Problem:** Tokenization of large files (>10K lines) is slow. Crabjar processes entire PR diffs which can be substantial.

**Request:**
- Optimize tokenization for large files
- Provide streaming/incremental tokenization API for diff processing
- Document performance characteristics and limits

### 5. Error Handling Improvements (P2)

**Problem:** Current error messages are minimal. Crabjar's JEV system needs to distinguish between:
- Syntax errors in input code
- Tokenizer internal errors
- Analyzer-specific failures

**Request:**
- Structured error types with context
- Recovery mode for partial analysis (best-effort results)
- Clear distinction between fatal and recoverable errors

### 6. Documentation Gaps (P2)

**Problem:** Crabjar developers need to understand:
- Which semantic analyzers exist and their exact behavior
- What constitutes "Propagated" vs "Recovered" vs "Panicked" error handling
- How mutability patterns are classified

**Request:**
- Comprehensive documentation for each semantic analyzer
- Examples of classification decisions (what triggers each category)
- Decision trees or flowcharts for complex analyzers

### 7. Integration Examples (P2)

**Problem:** Crabjar had to build its own integration examples from scratch.

**Request:**
- Include external integration examples in pesti repo
- Document patterns for using pesti with decision-making systems
- Provide reference implementation for PR review workflows

## Non-Requirements

- **Language support:** Pestis focuses on Rust; crabjar does not require multi-language support at this time
- **LLM embedding backend:** Crabjar uses pesti specifically for fast-path structural decisions without LLM dependency
- **Real-time analysis:** Batch processing of PR diffs is sufficient; streaming analysis is nice-to-have

## Success Criteria

Crabjar considers pesti integration production-ready when:
1. [x] pesti can be installed via pre-built binary or crates.io package (v0.1.4 published)
2. [ ] Semantic analyzer APIs are formally versioned and stable
3. [x] `analyze_ownership()` is implemented (detects owned/borrowed/mutably_borrowed patterns)
4. [ ] Tokenization of 10K-line files completes in <1 second
5. [ ] Error handling distinguishes syntax vs internal errors

## Timeline

- **Immediate (v0.12.x):** Requirements 1, 2 (binary distribution ✅, API stability in progress)
- **Near-term (v0.13.x):** Requirement 4 (performance at scale)
- **Long-term:** Requirements 5, 6, 7 (error handling, documentation, examples)

**Completed:** Requirements 1 and 3 are complete. pesti-structural-tokenizer v0.1.4 published to crates.io with complexity analysis module.

## Contact

For questions about crabjar's pesti integration requirements, contact the crabjar team via the crabjar repo issues.