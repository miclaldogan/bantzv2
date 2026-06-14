# Development workflow

A short guide to how changes land in Bantz. For environment setup, project
structure, and coding standards, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Branch strategy

**`main` is protected and always stable.** It's the single source of truth —
every commit on `main` should build, pass tests, and be deployable. Nothing is
pushed to `main` directly; changes arrive through reviewed pull requests.

```
main  ●───●───●───────────────●  (protected, stable)
            \                 /
   feat/123  ●───●───●───────●    (your branch → PR → review → merge)
```

## The loop

1. **Branch from `main`.** Keep it focused — one logical change per branch.
   - `feat/NN-short-description` — a new feature (NN = issue number)
   - `fix/NN-short-description` — a bug fix
   - `chore/short-description` — docs, CI, maintenance

   ```bash
   git switch main && git pull
   git switch -c fix/123-calendar-timezone
   ```

2. **Make your change** with tests. Run the suite locally:

   ```bash
   python -m pytest -q
   ```

3. **Open a pull request against `main`.** Describe what and why, and reference
   the issue (`Closes #123`). Keep PRs small — they're easier to review and
   safer to merge.

4. **Review.** At least one approving review is required before merge. CI runs
   on every PR. Pushing new commits dismisses prior approvals, so a fresh look
   always covers the final state.

5. **Merge to `main`.** Once approved and green, the PR is merged (squash keeps
   history tidy). The branch is then deleted.

## What `main` protection enforces

- ✅ Pull request required before merging — no direct pushes.
- ✅ At least **1 approving review**.
- ✅ **Stale reviews are dismissed** when new commits are pushed.
- 🚫 **No force pushes** to `main`.
- 🚫 **No deleting** `main`.

These rules keep history honest and `main` always releasable. (The repo
maintainer may use an admin merge for solo/urgent changes, but the default path
for everyone is: branch → PR → review → merge.)

## Good first issues

New here? Look for the `good first issue` / `help wanted` labels on the issue
tracker — great low-blast-radius places to start (intent routing, tool
reliability, UI polish). See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

Thanks for keeping `main` healthy. 🦌
