# ⚠️ SECURITY WARNING - IMMEDIATE ACTION REQUIRED

## Modal Credentials Exposed in Git History

**Date Detected:** 2025-11-19

### Compromised Credentials

The following Modal credentials were accidentally committed to git history (commit `2d6aba5`):

```
MODAL_TOKEN_ID: ak-2fPXyE2kShNW8RBPaLamY7
MODAL_TOKEN_SECRET: as-FKBZ1MK1id05l6BaBNiN6x
```

### IMMEDIATE ACTIONS REQUIRED

1. **Rotate Modal Credentials IMMEDIATELY**
   - Go to https://modal.com/settings/tokens
   - Revoke the compromised token: `ak-2fPXyE2kShNW8RBPaLamY7`
   - Generate new credentials
   - Update any systems using the old credentials

2. **Review Access Logs**
   - Check Modal.com access logs for any unauthorized usage
   - Look for unexpected deployments or function calls

3. **Monitor for Abuse**
   - Watch for unexpected charges on your Modal account
   - Set up billing alerts if not already configured

### Git History Note

While the files containing secrets have been deleted, **the secrets still exist in git commit history**.

To completely remove secrets from history (ADVANCED - can break things):
```bash
# Option 1: Using BFG Repo-Cleaner (recommended)
java -jar bfg.jar --delete-files modal_test_simple.py

# Option 2: Using git filter-branch (manual)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch modal_test_simple.py" \
  --prune-empty --tag-name-filter cat -- --all

# After either option:
git push origin --force --all
```

⚠️ **WARNING:** Force pushing rewrites history and can cause issues for other collaborators.

### Best Practices Going Forward

1. **Never commit credentials** to git
2. Use environment variables or secret management tools
3. Add secret files to `.gitignore` BEFORE committing
4. Use `git-secrets` or similar tools to prevent accidental commits
5. Rotate credentials regularly

### Status

- [x] Files deleted from working directory
- [x] Deletion committed to git
- [ ] **URGENT: Credentials rotated at modal.com**
- [ ] Git history cleaned (optional but recommended)

---

**Remember:** Deleting files doesn't remove them from git history. The old commits still contain the secrets until history is rewritten.
