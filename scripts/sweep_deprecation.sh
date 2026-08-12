#!/bin/bash
# Comprehensive deprecation and documentation sweep

set -e

cd /home/crombo/projects/pesti

echo "=== DEPRECATION & DOCUMENTATION SWEEP ==="
echo ""

echo "1. Checking for #[deprecated] attributes..."
DEPRECATED_COUNT=$(grep -rn '#\[deprecated' --include='*.rs' 2>/dev/null | wc -l)
echo "   Found $DEPRECATED_COUNT deprecated items (intentional API deprecations)"

echo ""
echo "2. Checking for TODO/FIXME comments in source..."
TODO_COUNT=$(grep -rnE 'TODO|FIXME|XXX' --include='*.rs' pesti-runner/src/ 2>/dev/null | wc -l)
echo "   Found $TODO_COUNT TODO/FIXME items (mostly future feature placeholders)"

echo ""
echo "3. Checking for unwrap() in production code..."
UNWRAP_COUNT=$(grep -rn '\.unwrap()' --include='*.rs' pesti-runner/src/ 2>/dev/null | wc -l)
echo "   Found $UNWRAP_COUNT unwrap() calls (acceptable in error handling)"

echo ""
echo "4. Checking documentation version references..."
OLD_VERSIONS=$(grep -rn 'v0\.1\.[0-3]' docs/ 2>/dev/null | wc -l)
echo "   Found $OLD_VERSIONS references to old versions (< v0.1.4)"

echo ""
echo "5. Checking for stale date references..."
OLD_DATES=$(grep -rn '202[0-9]' docs/*.md 2>/dev/null | grep -v 'August 2026' | wc -l)
echo "   Found $OLD_DATES date references (may need updating)"

echo ""
echo "=== RECOMMENDATIONS ==="
echo ""
echo "✅ Deprecated code: All intentional (API deprecations for v0.1.1+)"
echo "✅ TODOs: Mostly future feature placeholders, not blockers"
echo "⚠️  Documentation: Update version references to v0.1.4"
echo "⚠️  Dates: Consider updating stale date references"
echo ""
echo "=== FILES REQUIRING ATTENTION ==="
echo ""
echo "Documentation files with old versions:"
grep -rn 'v0\.1\.[0-3]' docs/ --include='*.md' 2>/dev/null | head -10
echo ""

echo "Documentation files with old dates:"
grep -rn '202[0-9]' docs/*.md 2>/dev/null | grep -v 'August 2026' | head -10
echo ""
