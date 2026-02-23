# Commit Convention — DeepSummit v2

This repo uses **Conventional Commits** prefixed with an emoji signal. Every commit in `main` should be readable in the git log at a glance — no "fix", "wip", or "update" commits.

---

## Format

```
<emoji> <type>(<scope>): <short description>

[optional body]
```

- **Short description**: imperative mood, lowercase, no period. "add rate limiting" not "Added rate limiting."
- **Scope**: the module or area changed. Examples: `api`, `ml`, `db`, `infra`, `frontend`, `ci`, `weather`
- **Body**: optional. Use it when the *why* isn't obvious from the title.

---

## Emoji Reference

| Emoji | Type | When to use |
|-------|------|-------------|
| ✨ | `feat` | New feature or capability |
| 🐛 | `fix` | Bug fix |
| ♻️ | `refactor` | Code restructure, no behaviour change |
| 🔥 | `chore` | Cleanup, removing dead code, formatting |
| 🧪 | `test` | Adding or fixing tests |
| 🚀 | `ci` | GitHub Actions, Cloud Build, deployment config |
| ⚡ | `perf` | Performance improvement |
| 📚 | `docs` | Documentation, README, guides, comments |
| 🗂️ | `data` | Dataset changes, enrichment, preprocessing output |

---

## Examples

```bash
# New features
✨ feat(api): add POST /predict endpoint with Pydantic validation
✨ feat(ml): implement unified multimodal transformer architecture
✨ feat(frontend): add CesiumJS globe with peak selection

# Bug fixes
🐛 fix(weather): handle Open-Meteo timeout with exponential backoff
🐛 fix(ml): correct attention mask shape for variable-length sequences

# Refactors
♻️ refactor(weather): extract Redis cache logic into WeatherCache class
♻️ refactor(api): split predict.py into service and handler layers

# Cleanup
🔥 chore(ml): remove unused v1 SAINT encoder imports
🔥 chore(frontend): remove unused shadcn components

# Tests
🧪 test(ml): add model accuracy and latency performance gates
🧪 test(api): add integration tests for /predict with mock inference service

# Deployment / CI
🚀 ci(github-actions): add ruff + mypy + pytest workflow on PR open
🚀 ci(cloud-build): add canary deployment with 10-min health check gate

# Performance
⚡ perf(inference): reduce feature tensor construction time by 40%
⚡ perf(weather): switch Redis serialisation from JSON to MessagePack

# Documentation
📚 docs(training-pipeline): add complete workflow guide with data flow diagrams
📚 docs(api): add OpenAPI schema and example requests to README

# Data
🗂️ data(peaks): enrich peaks_clean.csv with latitude/longitude coordinates
🗂️ data(training): regenerate features.csv after fixing experience feature calculation
```

---

## Branch → PR → Merge Flow

```
1. Branch off main:
   git checkout -b feature/phase1-db-schema

2. Work and commit with convention above.
   Each commit should represent one logical change.
   
3. Push and open a PR:
   - Title: mirrors your final commit message
   - Description: what changed, why, and what to look at in review
   - Reference roadmap phase: "Closes Phase 1: PostgreSQL schema migration"

4. CI must pass (ruff, mypy, pytest) before merge.

5. Merge via "Squash and merge" if the branch has noisy WIP commits.
   Use "Merge commit" if every commit on the branch is clean and meaningful.

6. Delete branch after merge.
```

---

## What Bad Commits Look Like (avoid these)

```bash
# ❌ Never
fix
update
wip
asdf
more changes
fixed the thing
trying something
```

If you're tempted to write one of these, the commit is probably too large or too vague. Break it into smaller pieces, or use the body to explain what's actually happening.

---

## Tip: Commit Often, Commit Small

A good commit is:
- One logical change
- Green (tests pass, linter passes)
- Describable in one line

If you can't describe it in one line, it's probably two commits.
