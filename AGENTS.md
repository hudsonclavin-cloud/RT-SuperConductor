# Repository contribution workflow

- Start each task from a fresh `origin/main`: fetch it, create a new work branch from it, and keep changes isolated to that branch.
- Before pushing, fetch `origin/main`, rebase the work branch onto it, and run `python -m pytest -q`.
- Never delete files or content under `docs/`. Preserve existing documentation when editing it.
- Keep one open question per pull request so review and follow-up stay focused.
- Hazel results for candidate compounds stay private: do not commit or push them (reports, `lambda` files, phonon files, Tc values, or text quoting them) until the owner says NC State's Office of Research Commercialization has reviewed them. Keep them under `data/raw/hazel/` (ignored). The Zr6Ga2Ni benchmark, an already-published compound, is exempt. Owner's decision of 2026-11-19; see `docs/hazel-candidates.md`.
