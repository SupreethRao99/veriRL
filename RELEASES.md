# Release Management

This project uses [towncrier](https://towncrier.readthedocs.io/) for automated changelog management and releases.

## Release Workflow

### 1. During Development

As you make changes, create changelog fragments:

```bash
# Create a feature fragment
echo "Added new task type to environment" > .changelog.d/123.feature.md

# Create a bug fix fragment
echo "Fixed STDOUT format parsing error" > .changelog.d/124.bugfix.md

# Create a documentation fragment
echo "Updated README with new API documentation" > .changelog.d/125.doc.md

# Create a miscellaneous fragment (no details in changelog)
echo "Code cleanup and refactoring" > .changelog.d/126.misc.md
```

**Fragment format:**
- File name: `.changelog.d/<ID>.<TYPE>.md`
- `<ID>`: PR number, commit hash, or any unique identifier
- `<TYPE>`: `feature`, `bugfix`, `doc`, or `misc`
- Content: One-line description of the change

### 2. Before Release

When ready to release, use the GitHub Actions workflow:

1. Go to: https://github.com/SupreethRao99/veriRL/actions
2. Click "Release" workflow
3. Click "Run workflow"
4. Enter version (e.g., `0.2.1`)
5. Workflow will:
   - Run `towncrier build --version 0.2.1`
   - Update CHANGELOG.md
   - Update pyproject.toml version
   - Commit changes
   - Create git tag `v0.2.1`
   - Push to GitHub
   - Create GitHub Release with changelog

### 3. Manual Release (if workflow fails)

```bash
# Install towncrier
uv pip install towncrier

# Generate changelog and update version
towncrier build --version 0.2.1

# Review changes
git status

# Commit
git add CHANGELOG.md pyproject.toml .changelog.d/
git commit -m "Release v0.2.1"

# Create tag
git tag -a v0.2.1 -m "Release v0.2.1"

# Push
git push origin main v0.2.1

# Create release on GitHub (via web UI or gh CLI)
gh release create v0.2.1 --title "v0.2.1" --notes-file CHANGELOG.md
```

## Version Numbering

Uses [Semantic Versioning](https://semver.org/):

- **MAJOR** (0.X.0): Breaking changes, major architecture updates
- **MINOR** (0.1.X): New features, new tasks, significant enhancements
- **PATCH** (0.1.1): Bug fixes, documentation updates, minor improvements

Examples:
- Initial release: `0.1.0`
- Add new task: `0.2.0` (minor bump)
- Fix bug: `0.2.1` (patch bump)
- Breaking change: `1.0.0` (major bump)

## Viewing Releases

- **Changelog:** https://github.com/SupreethRao99/veriRL/blob/main/CHANGELOG.md
- **Releases:** https://github.com/SupreethRao99/veriRL/releases
- **Tags:** https://github.com/SupreethRao99/veriRL/tags

## Fragment Types Reference

### feature
New functionality, new tasks, new endpoints, new capabilities.

Example: `0001.feature.md`
```
Added AXI-Stream FIFO task with 34 test assertions
```

### bugfix
Bug fixes, corrections, error handling improvements.

Example: `0002.bugfix.md`
```
Fixed STDOUT format to remove extraneous validation output
```

### doc
Documentation updates, README changes, new guides.

Example: `0003.doc.md`
```
Updated deployment guide with GitHub Actions instructions
```

### misc
Internal changes, refactoring, code cleanup (not shown in changelog).

Example: `0004.misc.md`
```
Refactor environment initialization for clarity
```

## Changelog Format

Generated CHANGELOG.md follows this structure:

```markdown
## [0.2.0] - 2026-04-04

### Features
- Feature 1 (PR #123)
- Feature 2 (PR #124)

### Bug Fixes
- Fix 1 (PR #125)
- Fix 2 (PR #126)

### Documentation
- Doc update 1 (PR #127)
- Doc update 2 (PR #128)
```

## Tips

1. **Create fragments frequently** — one per PR or significant change
2. **Use clear descriptions** — fragments become the changelog, so be descriptive
3. **Keep it user-facing** — describe what users/judges care about
4. **Test release locally** — run `towncrier build --draft` to preview
5. **One release per version** — don't release twice for the same version

## Troubleshooting

### Fragment not appearing in changelog
- Check file name format: `.changelog.d/<ID>.<TYPE>.md`
- Verify `<TYPE>` is one of: feature, bugfix, doc, misc
- Ensure file is in `.changelog.d/` directory

### Changelog build fails
- Check pyproject.toml `[tool.towncrier]` configuration
- Verify all fragments have valid type
- Run with `--draft` flag to preview without modifying files

### Version not updating
- Check pyproject.toml has `version = "0.1.0"` field
- Towncrier auto-updates this during `build` command

## References

- [Towncrier Docs](https://towncrier.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
