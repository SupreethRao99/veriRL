# Repository Setup Guide

This document explains how to enable all the automated workflows for this repository.

## Prerequisites

Before enabling workflows, ensure these secrets are configured:

### 1. HF_TOKEN (Required for Deployment)

1. Go to https://huggingface.co/settings/tokens
2. Create a new API token with **write** permission for spaces
3. Go to repository settings: https://github.com/SupreethRao99/veriRL/settings/secrets/actions
4. Click "New repository secret"
5. Name: `HF_TOKEN`
6. Value: Your HF API token
7. Save

### 2. GITHUB_TOKEN (Automatic)

GitHub provides this automatically — no action needed.

## Step 1: Enable Branch Protection

1. Go to repository settings: https://github.com/SupreethRao99/veriRL/settings/branches

2. Click "Add rule"

3. **Branch name pattern:** `main`

4. **Basic Settings:**
   - [x] Require a pull request before merging
   - [x] Require approvals (set to 1)
   - [x] Dismiss stale pull request approvals when new commits are pushed

5. **Status Checks:**
   - [x] Require status checks to pass before merging
   - [x] Require branches to be up to date before merging
   
   Select these checks:
   - `validate` (from ci-cd.yml)
   - `docker-build` (from ci-cd.yml)

6. **Restrictions:**
   - [ ] Allow force pushes (UNCHECK)
   - [ ] Allow deletions (UNCHECK)

7. Click "Create"

## Step 2: Verify Workflows

1. Go to https://github.com/SupreethRao99/veriRL/actions

2. Verify these workflows exist:
   - ✓ `.github/workflows/ci-cd.yml` (auto on push/PR)
   - ✓ `.github/workflows/release.yml` (manual)
   - ✓ `.github/workflows/validate.yml` (PR checks)

## Step 3: Enable Container Registry Access

GitHub Container Registry (ghcr.io) is enabled by default.

Docker images will be pushed to:
- **On merge to main:** `ghcr.io/SupreethRao99/veriRL:main`
- **On release:** `ghcr.io/SupreethRao99/veriRL:v0.3.0` + `:latest`

## Complete Workflow

### Development → Merge → Deploy → Release

```
┌─────────────────────┐
│  Feature Branch     │
│  - Make changes     │
│  - Add fragment     │
│  - Push to fork     │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────────────────┐
│  Pull Request to main           │
│  - CI validates (auto)          │
│  - Docker builds (auto)         │
│  - Status checks block merge if │
│    validate or build fails      │
└──────────┬──────────────────────┘
           │
           ↓ (on approval + checks pass)
┌──────────────────────────────────┐
│  Merge to main                   │
│  - openenv validate              │
│  - Docker build                  │
│  - Push image: ghcr.io:.../main  │
│  - Deploy to HF Spaces           │
│  - Traceable by commit SHA       │
└──────────┬───────────────────────┘
           │
           ↓ (optional - when ready)
┌──────────────────────────────────────┐
│  Release (Manual)                    │
│  Actions → Release → Run workflow    │
│  - Enter version: 0.3.0              │
│  - Generate CHANGELOG.md             │
│  - Update pyproject.toml             │
│  - Create git tag v0.3.0             │
│  - Build Docker image                │
│  - Push: ghcr.io:.../v0.3.0          │
│  - Push: ghcr.io:.../latest          │
│  - Publish GitHub Release            │
│  - Traceable by version number       │
└──────────────────────────────────────┘
```

## Usage After Setup

### For Developers

```bash
# 1. Create feature branch
git checkout -b feature/my-change

# 2. Make changes
vim server/verirl_env_environment.py
vim tests/test_environment.py

# 3. Create changelog fragment
echo "Added new feature" > .changelog.d/123.feature.md

# 4. Commit and push
git add -A
git commit -m "Add new feature"
git push origin feature/my-change

# 5. Create PR on GitHub
# Go to https://github.com/SupreethRao99/veriRL/pulls
# Click "New pull request"
# Select: main ← feature/my-change

# 6. Wait for CI checks (auto-run)
# All checks must pass before merge is allowed

# 7. Merge via GitHub UI
# Merge to main when approved + checks pass
```

### For Releases

```bash
# 1. Go to Actions → Release
# 2. Click "Run workflow"
# 3. Enter version: 0.3.0
# 4. Wait for workflow to complete

# Result:
# - GitHub Release: https://github.com/SupreethRao99/veriRL/releases/tag/v0.3.0
# - Docker image: ghcr.io/SupreethRao99/veriRL:v0.3.0
# - Latest tag: ghcr.io/SupreethRao99/veriRL:latest
# - Changelog in CHANGELOG.md
```

## Verifying Setup

### Test Branch Protection

```bash
# This should fail with "Repository does not allow force pushes"
git push --force origin main
```

### Test PR Workflow

```bash
# Create test branch
git checkout -b test/setup-verification

# Add a dummy file
echo "test" > test.txt

# Commit and push
git add test.txt
git commit -m "Test CI workflow"
git push origin test/setup-verification

# Create PR on GitHub
# Watch Actions tab to verify:
# ✓ validate job runs
# ✓ docker-build job runs
# Both must pass
```

### Test Docker Image Push

On next merge to main, verify Docker image is pushed:

```bash
# After merge to main, wait ~10 minutes
# Then check:
docker pull ghcr.io/SupreethRao99/veriRL:main
docker run -p 8000:8000 ghcr.io/SupreethRao99/veriRL:main

# Verify space is live at:
# https://huggingface.co/spaces/Supreeth/verirl-env
```

## Troubleshooting

### "Cannot push to main"
✓ Expected — branch protection is working
→ Create a feature branch instead

### "HF_TOKEN not found"
- Check secret is configured at settings/secrets/actions
- Verify name is exactly `HF_TOKEN` (case-sensitive)
- Regenerate token and update secret

### "Docker build fails in CI"
- Run locally: `docker build -f server/Dockerfile .`
- Fix issue
- Push again — CI re-runs automatically

### "Validation fails in CI"
- Run locally: `openenv validate`
- Fix issue
- Push again — CI re-runs automatically

## Maintenance

### Regular Checks

Monthly (recommended):
- Check GitHub Actions for failures
- Review Docker image sizes
- Verify HF Spaces is responding

Quarterly:
- Review and update dependencies
- Check for security updates
- Rotate tokens if needed

## Documentation

- **Workflow:** [CONTRIBUTING.md](../../CONTRIBUTING.md)
- **Branch Protection:** [BRANCH_PROTECTION.md](BRANCH_PROTECTION.md)
- **Releases:** [RELEASES.md](../../RELEASES.md)
- **Deployment:** [DEPLOYMENT.md](DEPLOYMENT.md)

---

**Setup is complete!** Your repository now has professional-grade CI/CD with full version tracking. 🚀
