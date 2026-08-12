# Release Process

## Versioning Strategy

PESTI follows [Semantic Versioning (SemVer)](https://semver.org/) with the following guidelines:

- **MAJOR** (0.x.0): Breaking API changes, significant refactors
- **MINOR** (x.y.0): New features, backwards-compatible additions
- **PATCH** (x.y.z): Bug fixes, internal improvements

## Current Version

**v0.1.1** - Initial production release with pure Rust dequantization

## Release Checklist

### Pre-Release
- [ ] All tests passing (`cargo test --workspace`)
- [ ] Clippy warnings resolved (`cargo clippy --all-features`)
- [ ] Code formatted (`cargo fmt`)
- [ ] Documentation updated
- [ ] CHANGELOG.md generated

### Release Steps

#### Option A: Automated (GitHub Actions)
1. Push to `main` branch → CI runs automatically
2. Trigger "Release & Version Bump" workflow manually
3. Select version bump type: `patch`, `minor`, or `major`
4. Workflow updates versions and creates tag

#### Option B: Manual
```bash
# 1. Update version in root Cargo.toml
sed -i 's/^version = "[0-9]\+\.[0-9]\+\.[0-9]\+"/version = "0.1.1"/g' Cargo.toml

# 2. Update individual package versions (if not workspace-managed)
for TOML in */Cargo.toml; do
  if grep -q "version.workspace = true" "$TOML"; then
    continue
  fi
  sed -i 's/^version = "[0-9]\+\.[0-9]\+\.[0-9]\+"/version = "0.1.1"/g' "$TOML"
done

# 3. Generate changelog
conventional-changelog -p angular -r 0 > CHANGELOG.md

# 4. Commit and tag
git add Cargo.toml */Cargo.toml CHANGELOG.md
git commit -m "chore: bump version to 0.1.1"
git tag v0.1.1
git push
git push origin v0.1.1
```

### Post-Release
- [ ] Verify GitHub Actions workflow ran successfully
- [ ] Check CHANGELOG.md for completeness
- [ ] Announce release (if applicable)

## Breaking Changes in 0.x

Since we're in pre-1.0, breaking changes are expected. They will be:
- Documented in CHANGELOG.md under "Breaking Changes"
- Accompanied by migration notes where possible
- Bumped as MAJOR version (0.x → 0.y)

## Example Release Flow

```bash
# Developer workflow
git checkout main
git pull origin main
cargo test --workspace
cargo clippy --all-features
cargo fmt

# When ready to release
nano RELEASE.md  # Update changelog entries
git add CHANGELOG.md
git commit -m "docs: update CHANGELOG for v0.1.1"

# Trigger automated release (via GitHub UI or CLI)
gh workflow run release.yml -f version_bump=patch
```

## Version Bump Guidelines

| Change Type | Example | Bump Type |
|-------------|---------|-----------|
| Bug fix in dequantization | Fixed Q4_0 dequant error | `patch` (0.1.0 → 0.1.1) |
| New feature | Added Q8_0 support | `minor` (0.1.0 → 0.2.0) |
| Breaking API change | Removed old FFI layer | `major` (0.1.0 → 1.0.0) |

## Notes

- Workspace-managed versions use `version.workspace = true` in package Cargo.toml files
- Root `Cargo.toml` defines the canonical version
- All workspace members inherit from root unless explicitly overridden
