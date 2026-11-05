# Repository contribution workflow

- Start each task from a fresh `origin/main`: fetch it, create a new work branch from it, and keep changes isolated to that branch.
- Before pushing, fetch `origin/main`, rebase the work branch onto it, and run `python -m unittest discover -s tests`.
- Never delete files or content under `docs/`. Preserve existing documentation when editing it.
- Keep one open question per pull request so review and follow-up stay focused.
