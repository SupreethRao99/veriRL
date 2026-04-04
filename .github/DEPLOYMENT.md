# Deployment Guide

This repository includes GitHub Actions workflows for automated validation and deployment to Hugging Face Spaces.

## Setup

### 1. Create HF_TOKEN Secret

1. Go to https://github.com/SupreethRao99/veriRL/settings/secrets/actions
2. Click "New repository secret"
3. Name: `HF_TOKEN`
4. Value: Your Hugging Face API token (from https://huggingface.co/settings/tokens)
   - Token must have write access to spaces
5. Click "Add secret"

## Workflows

### CI/CD Pipeline (`.github/workflows/ci-cd.yml`)

**Triggers:** Every push to `main` (automatic)

**Three jobs run in sequence:**
1. **Validate** — `openenv validate` (spec compliance)
2. **Docker Build** — builds Docker image (Dockerfile validation)
3. **Deploy** — runs `openenv push` to HF Spaces (only if 1 & 2 pass)

**Status:** Visible in repo Actions tab. Green checkmark = all passed & deployed.

### Validate Workflow (`.github/workflows/validate.yml`)

**Triggers:** Pull requests and manual pushes to branches

**What it does:**
1. Installs dependencies via `uv sync`
2. Runs `openenv validate` — ensures spec compliance
3. Builds Docker image — verifies Dockerfile works

**Status:** Visible as PR checks

### Deploy Workflow (`.github/workflows/deploy.yml`) — Optional

**Triggers:** Manual workflow_dispatch (for custom deployment)

**When to use:** If you want to deploy with a different `repo_id` or to a private space

**Steps:**
1. Go to https://github.com/SupreethRao99/veriRL/actions
2. Click "Deploy to HF Spaces"
3. Click "Run workflow"
4. Fill in inputs:
   - **repo_id**: `your-org/your-space`
   - **private**: `true` or `false`
5. Click "Run workflow"

## Automatic Deployment

Once you have `HF_TOKEN` secret configured, deployment is **automatic**:

1. **Every push to main triggers CI/CD:**
   - Validates environment ✓
   - Builds Docker image ✓
   - Deploys to HF Spaces ✓

2. **Check deployment status:**
   - Go to https://github.com/SupreethRao99/veriRL/actions
   - Click latest "CI/CD" workflow run
   - All 3 jobs should be green (✓)

3. **Your space is live at:**
   ```
   https://huggingface.co/spaces/Supreeth/verirl-env
   ```

## Local Deployment (Backup)

If automated deployment fails:

```bash
export HF_TOKEN=<your_hf_token>
openenv push --repo-id Supreeth/verirl-env
```

## Troubleshooting

### "HF_TOKEN secret not found"
- ✓ Create the secret in repo settings (see Setup above)
- ✓ Restart the workflow run

### "openenv push failed"
- Check HF_TOKEN has write permission: https://huggingface.co/settings/tokens
- Verify repo_id exists or is allowed for your account
- Check GitHub Actions logs for detailed error

### "Docker build failed"
- Run locally: `docker build -t verirl:latest -f server/Dockerfile .`
- Fix errors in Dockerfile or dependencies
- Push fixes to main (workflow will re-run)

## Validation Before Submission

1. **Ensure validation passes:**
   ```
   ✓ openenv validate
   ✓ Docker build succeeds
   ```

2. **Test space is live:**
   ```bash
   curl -X POST https://Supreeth.hf.space/reset \
     -H "Content-Type: application/json" -d '{}'
   # Should return HTTP 200
   ```

3. **Run baseline inference:**
   ```bash
   export ENV_BASE_URL=https://Supreeth.hf.space
   python inference.py
   # Should produce valid [START]/[STEP]/[END] output
   ```

4. **Run validation script:**
   ```bash
   ./validation.sh https://Supreeth.hf.space
   ```
