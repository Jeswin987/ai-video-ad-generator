# CLAUDE.md — ai-video-ad-generator

## Context
- Team-of-3 course mini-project (M.Sc.). I (Jesvin) am polishing it for my portfolio.
  Never describe it as solo work or attribute teammates' work to me.
- Windows 11, Git Bash. GPU: RTX 2050, 4 GB VRAM. Default branch: main.
- Python: always use venv/Scripts/python.exe (Python 3.13, CUDA PyTorch).
  Run tests with: venv/Scripts/python.exe -m pytest

## Git
- Never commit to main. Work only on the branch I name.
- One scoped commit per item. Stage files by name only. Never use
  `git add -A`, `git add .`, `git add -u`, or `git commit -a`.
- Before each commit, run the full test suite. If no tests exist yet,
  tell me; never skip silently or claim a pass.
- After each commit, stop and wait for my "go".
- Never rewrite history: no force-push, no rebase or amend of pushed
  commits, no filter-repo. Never change the author of existing commits.
- Never push. I push from my own terminal. Keep Co-Authored-By trailers.

## Tests
- Tests must not load real models, need the GPU, or download anything.
- Tests must not write into models/, output/, logs/, or temp/. Use pytest's tmp_path.
- For real bugs: write a failing test first, then fix.

## Secrets
- The Hugging Face token is stored in ~/.cache/huggingface/. Never read,
  print, or copy anything from that folder, and never read any .env file.
- Never put tokens in code, commands, logs, or commit messages.

## Accuracy
- Every claim and number in docs must trace to code, config, or a committed
  log. If it can't, mark it CAN'T VERIFY. Never estimate or round to fill gaps.
- No licence: don't add a LICENSE file or licence text.

## Heavy operations: ask me first
- Downloading models, installing packages, running the pipeline, or using the GPU.
- Deleting anything outside the repo, or anything in models/ or output/.