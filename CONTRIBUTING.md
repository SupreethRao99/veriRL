# Contributing Guide

Thank you for contributing to VeriRL! This guide explains the development workflow.

## Workflow Overview

```
1. Create feature branch from main
2. Make changes
3. Create changelog fragment (.changelog.d/<ID>.{feature,bugfix,doc,misc}.md)
4. Push to your fork
5. Create pull request to main
6. CI/CD runs (validate, docker build) on PR
7. Review & merge to main
8. CI/CD auto-deploys to HF Spaces
9. Create release via GitHub Actions (bumps version, publishes Docker image)
```

## Branch Protection Rules

The `main` branch is protected:
- ✓ **No direct pushes** — all changes via pull requests
- ✓ **Requires status checks** — openenv validate + docker build must pass
- ✓ **Requires reviews** — at least 1 approval recommended
- ✓ **Requires updated branch** — must be up-to-date with main

## Step-by-Step Guide

### 1. Set Up Local Environment

```bash
# Clone the repo
git clone https://github.com/SupreethRao99/veriRL.git
cd veriRL

# Create feature branch
git checkout -b feature/your-feature-name

# Install dependencies
uv sync

# Verify environment works
openenv validate
```

### 2. Make Changes

```bash
# Edit files, add features, fix bugs
vim server/verirl_env_environment.py
vim tests/test_environment.py

# Test locally
pytest
openenv validate
docker build -t verirl:test -f server/Dockerfile .
```

### 3. Create Changelog Fragment

For EVERY change, create a fragment:

```bash
# Feature
echo "Added new task: XYZ accelerator" > .changelog.d/pr-123.feature.md

# Bug fix
echo "Fixed STDOUT format parsing" > .changelog.d/pr-124.bugfix.md

# Documentation
echo "Updated README with new examples" > .changelog.d/pr-125.doc.md

# Internal (not shown in changelog)
echo "Refactored evaluator for clarity" > .changelog.d/pr-126.misc.md
```

Fragment format:
- File: `.changelog.d/<ID>.<TYPE>.md`
- ID: PR number, commit hash, or unique identifier
- TYPE: feature, bugfix, doc, misc
- Content: One-line description

### 4. Commit & Push

```bash
# Stage changes
git add server/ tests/ .changelog.d/

# Commit with clear message
git commit -m "Add new task and update documentation

- Implement XYZ accelerator task
- Add 25 test cases
- Update README with task description
- Create changelog fragment"

# Push to your fork
git push origin feature/your-feature-name
```

### 5. Create Pull Request

1. Go to https://github.com/SupreethRao99/veriRL/pulls
2. Click "New pull request"
3. Select: `main` ← `feature/your-feature-name`
4. Fill in description:
   ```markdown
   ## Summary
   Add XYZ accelerator task to environment

   ## Changes
   - New task with 25 assertions
   - Updated scoring weights
   - Added reference implementation
   
   ## Testing
   - [x] openenv validate passes
   - [x] Docker builds successfully
   - [x] All tests pass locally
   ```
5. Submit PR

### 6. CI/CD Validation (Automatic)

When you create the PR, GitHub Actions automatically:
- ✓ Validates openenv spec
- ✓ Builds Docker image
- ✓ Runs tests

**If checks fail:**
1. Review error logs in Actions tab
2. Fix issues locally
3. Commit and push again
4. Checks re-run automatically

### 7. Review & Merge

- Code review (recommended)
- Request changes if needed
- Merge to main when ready

```bash
# GitHub UI: Click "Merge pull request"
# Or via CLI:
gh pr merge <PR_NUMBER>
```

### 8. Auto-Deploy (On Merge)

When you merge to main, GitHub Actions automatically:
1. Validates environment
2. Builds Docker image
3. Pushes image to `ghcr.io` as `main` tag
4. Deploys to HF Spaces

**Done!** Your changes are live.

### 9. Create Release (Optional)

When ready to cut a release:

1. Go to https://github.com/SupreethRao99/veriRL/actions
2. Click "Release" workflow
3. Click "Run workflow"
4. Enter version (e.g., `0.3.0`)
5. Workflow:
   - Generates CHANGELOG.md from fragments
   - Updates version in pyproject.toml
   - Creates git tag `v0.3.0`
   - Builds & pushes Docker image to `ghcr.io/SupreethRao99/veriRL:v0.3.0`
   - Creates GitHub Release

**Done!** Release is published with versioned Docker image.

## Naming Conventions

### Branch Names
```
feature/<name>      New feature or task
bugfix/<name>       Bug fix
docs/<name>         Documentation update
refactor/<name>     Code refactoring
```

Examples:
```
feature/axi-fifo-task
bugfix/stdout-format
docs/deployment-guide
refactor/evaluator
```

### Commit Messages
```
<type>: <subject>

<body>

<footer>
```

Types: feat, fix, docs, refactor, test, chore

Example:
```
feat: Add AXI-Stream FIFO task

- Implement 4-entry FIFO with backpressure
- Add 34 test assertions
- Update README with task description

Fixes #123
```

## Testing Locally

Before pushing:

```bash
# Run tests
pytest

# Check code style (if configured)
# pycodestyle, black, ruff, etc.

# Validate environment
openenv validate

# Build Docker image
docker build -t verirl:test -f server/Dockerfile .

# Test inference (if server is running)
export ENV_BASE_URL=http://localhost:8000
python inference.py
```

## Common Issues

### "Validation failed on PR"
1. Check error message in Actions tab
2. Run locally: `openenv validate`
3. Fix issue
4. Push again

### "Docker build failed"
1. Check Dockerfile
2. Build locally: `docker build -f server/Dockerfile .`
3. Fix and commit
4. Re-run workflow

### "Can't push to main"
- Branch protection is enabled
- Create PR from feature branch instead
- Merge via pull request

### "Forgot changelog fragment"
1. Create fragment: `.changelog.d/<ID>.<TYPE>.md`
2. Commit: `git add .changelog.d/ && git commit -m "Add changelog fragment"`
3. Push
4. Checks re-run

## Version Numbers

Uses Semantic Versioning: `MAJOR.MINOR.PATCH`

- `0.1.0` → `0.2.0` = new feature/task (minor bump)
- `0.2.0` → `0.2.1` = bug fix (patch bump)
- `0.2.1` → `1.0.0` = breaking change (major bump)

## Docker Images

Every merge to main publishes an image:
```
ghcr.io/SupreethRao99/veriRL:main
```

Every release publishes versioned images:
```
ghcr.io/SupreethRao99/veriRL:v0.2.0
ghcr.io/SupreethRao99/veriRL:latest
```

Pull and run:
```bash
docker pull ghcr.io/SupreethRao99/veriRL:v0.2.0
docker run -p 8000:8000 ghcr.io/SupreethRao99/veriRL:v0.2.0
```

## Questions?

- Check existing issues: https://github.com/SupreethRao99/veriRL/issues
- Review past PRs: https://github.com/SupreethRao99/veriRL/pulls
- Check RELEASES.md for release management details

---

**Thank you for contributing!** 🚀
