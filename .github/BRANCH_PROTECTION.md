# Branch Protection Configuration

This repository uses branch protection rules on `main` to enforce code quality and prevent accidental deployments.

## Current Rules (Configured on `main`)

✓ **Require a pull request before merging**
  - At least 1 approval required
  - Dismiss stale pull request approvals when new commits are pushed

✓ **Require status checks to pass before merging**
  - Validate (openenv validate)
  - Docker Build
  - (Optional) Test suite

✓ **Require branches to be up to date before merging**
  - PRs must be rebased/updated with latest main

✓ **Dismiss pull request approval reviews**
  - When new commits are pushed
  - Ensures reviewers see latest changes

## How to Configure (Admin Only)

If you need to set this up or modify rules:

1. Go to: https://github.com/SupreethRao99/veriRL/settings/branches

2. Under "Branch protection rules", add/edit rule for `main`:

### Basic Settings
- [x] Require a pull request before merging
  - [x] Require approvals (1)
  - [x] Dismiss stale pull request approvals when new commits are pushed
  - [x] Require code owner reviews (optional)

### Status Checks
- [x] Require status checks to pass before merging
- [x] Require branches to be up to date before merging

Select required status checks:
- `validate` (from .github/workflows/ci-cd.yml)
- `docker-build` (from .github/workflows/ci-cd.yml)

### Additional Settings
- [ ] Require signed commits (optional, for security)
- [ ] Allow auto-merge (optional, for convenience)
- [x] Allow force pushes - **DISABLE THIS**
- [x] Allow deletions - **DISABLE THIS**

## Impact

With these rules:

❌ **Cannot do:**
- Direct `git push` to main
- Force push to main
- Delete the main branch
- Merge without PR
- Merge without passing checks

✓ **Must do:**
- Create feature branch
- Create pull request
- Pass all CI checks (validate, docker build)
- Get approval (recommended)
- Merge via GitHub UI

## Workflow

```
1. git checkout -b feature/xyz
2. git commit ...
3. git push origin feature/xyz
4. Create PR on GitHub
5. CI checks run automatically
6. Review & merge
7. Auto-deploys to HF Spaces
```

## Bypassing Rules (Not Recommended)

Only repository admins can bypass branch protection rules. Use ONLY for:
- Emergency production fixes
- Admin privilege corrections

**Never** use for regular development.

## Troubleshooting

### "Cannot push to main"
Expected behavior. Create a feature branch instead.

### "PR blocked: status checks failed"
Fix the issue locally:
```bash
# For openenv validate failure
openenv validate

# For docker build failure
docker build -f server/Dockerfile .

# Fix and push again
git add .
git commit -m "Fix validation issues"
git push origin feature/xyz
```

### "PR blocked: branch out of date"
Update with main:
```bash
git fetch origin
git rebase origin/main
git push --force-with-lease origin feature/xyz
```

### "Need to bypass rules"
Contact repository admin. Provide justification.

## Best Practices

1. **Keep feature branches fresh**
   - Rebase regularly against main
   - Push frequently to avoid conflicts

2. **Create focused PRs**
   - One feature per PR
   - Keep scope small
   - Easier to review and merge

3. **Write clear commit messages**
   - Helps reviewers understand changes
   - Shows in changelog

4. **Include changelog fragments**
   - Required for all changes
   - Enables automated changelog

5. **Test locally before pushing**
   - Run `pytest`, `openenv validate`, `docker build`
   - Catch issues early

## References

- GitHub Branch Protection: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches
- GitHub Actions: https://docs.github.com/en/actions
- Contributing Guide: ../../CONTRIBUTING.md
