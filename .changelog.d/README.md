# Changelog Fragments

This directory contains news fragments for the upcoming release. Each file represents a change that should appear in the changelog.

## Fragment Format

Create a file named: `<PR_NUMBER>.<TYPE>.md`

Where:
- `<PR_NUMBER>` = GitHub PR/commit number or any unique identifier
- `<TYPE>` = one of: `feature`, `bugfix`, `doc`, `misc`

## Examples

### New Feature
**File:** `.changelog.d/123.feature.md`
```markdown
Added task enumeration and automatic grader validation to inference script
```

### Bug Fix
**File:** `.changelog.d/456.bugfix.md`
```markdown
Fixed STDOUT format to strictly match rubric specification (no extraneous output)
```

### Documentation
**File:** `.changelog.d/789.doc.md`
```markdown
Updated README with deployment instructions and baseline scores
```

## Generating Changelog

Before a release, run:
```bash
towncrier build --version 0.2.0
```

This will:
1. Combine all fragments into `CHANGELOG.md`
2. Remove fragment files
3. Update version in `pyproject.toml`
4. Create a commit

Then create a GitHub release with the changelog.
