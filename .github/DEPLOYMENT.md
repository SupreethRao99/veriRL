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

### Validate Workflow (`.github/workflows/validate.yml`)

**Triggers:** Every push to `main` and pull requests

**What it does:**
1. Installs dependencies via `uv sync`
2. Runs `openenv validate` — ensures spec compliance
3. Builds Docker image — verifies Dockerfile works

**Status:** Visible in repo "Actions" tab or PR checks

### Deploy Workflow (`.github/workflows/deploy.yml`)

**Triggers:** Manual (workflow_dispatch)

**What it does:**
1. Validates environment
2. Builds Docker image
3. Runs `openenv push` to deploy to HF Spaces
4. Prints deployment URL

## Deploying to HF Spaces

### Option 1: Manual Trigger (Recommended for Testing)

1. Go to https://github.com/SupreethRao99/veriRL/actions
2. Click "Deploy to HF Spaces" workflow
3. Click "Run workflow"
4. Fill in inputs:
   - **repo_id**: `Supreeth/verirl-env` (your target space)
   - **private**: `false` (or `true` for private spaces)
5. Click "Run workflow"
6. Wait for completion (check logs)
7. Space URL will be printed: `https://huggingface.co/spaces/Supreeth/verirl-env`

### Option 2: Local Deployment

If CI/CD deployment fails, deploy locally:

```bash
export HF_TOKEN=<your_hf_token>
openenv push --repo-id Supreeth/verirl-env
# or for private space
openenv push --repo-id Supreeth/verirl-env --private
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
