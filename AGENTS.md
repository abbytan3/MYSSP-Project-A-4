# AGENTS.md: rules for every coding agent in this repo

Read this whole file before doing anything. It applies to every session, whoever runs it and whichever agent they use (Claude Code reads it through `CLAUDE.md`).

## What this project is

MYSSP 2026, project A4: "Can Off-the-Shelf Tampering Detectors Find AI Edits in Bank Statements?" Team: Abby, Alycia, Nazhan. Mentor and co-author: Jia Yu.

We test whether pre-trained document-tampering detectors (ADCD-Net, DTD, DAF) catch AI text edits (AnyText, UDiffText, TextCtrl) and handcrafted edits on real Malaysian bank statements. Results are counted per document and per edited location, not per pixel.

- Detectors run on the Modal cloud GPU. Code is written and tested on this laptop.
- The bank statements are under NDA. They never enter this repo or anything public (see NDA rules).
- **`docs/PLAN.md` is the confirmed experiment plan.** Read it before any experiment task, and build exactly what it says. Never edit it; if it seems wrong, stop and ask.
- Task IDs such as T14 refer to the Tasks tab of the team's command centre sheet. The person tells you the ID and what the task says.
- Deadline: Fri 20 Nov 2026, 11 PM (Malaysia time).

If this file and another file disagree, stop and ask.

## First steps in every session

1. Ask who you're working with (Abby, Alycia or Nazhan) and the task ID, unless they've said.
2. Explain the task in 3 to 5 plain sentences before any work: what it does, why the project needs it, what comes before it and what uses its output next. Read `docs/PLAN.md` first if the task is an experiment, so the explanation is right.
3. Summarise these rules in five lines so the person knows you read them.
4. Run `git config user.name` and `git config user.email`. If either is empty or not the person's own GitHub details, stop and ask them to fix it (`docs/SETUP.md`, step 1). Never change it yourself.
5. Run `git fetch origin` and `git status`. If this copy is behind `origin/main`, bring it up to date before changing anything (see Git rules).
6. Check the tools the task needs exist (`python3 --version` or `python --version`, `modal --version`). Never run `modal token` commands: the token is private.
7. If the task touches bank-statement pages, ask the person to confirm "Help improve Claude" is switched off on their account (claude.ai → Settings → Privacy). Don't open any page until they confirm.
8. For experiment tasks, read `docs/PLAN.md`.
9. List the files this task will change and tell the person before you start.

## Who owns which folder

| Folder | Holds | Who changes it |
|---|---|---|
| `shared/` | Sampling, upload to Modal, output format, scoring script | Abby |
| `analysis/human/` | Stage 2 ratings: people vs detectors | Owner set on the command centre |
| `analysis/ablation/` | Week 7 rescoring | Abby |
| `detectors/adcd/` | ADCD-Net Modal job | Abby |
| `detectors/dtd/` | DTD Modal job | Alycia |
| `detectors/daf/` | DAF Modal job | Nazhan |
| `results/<detector>/` | Score tables and figures from that detector's runs | Whoever ran the run |
| `results/combined/` | Cross-detector tables and figures | Abby |
| `docs/` | Setup guide, plan, decisions log, write-up notes | Anyone |
| `prior_work/abby/` | Abby's Weeks 4–7 code (as run) | Abby |
| `prior_work/alycia/` | Alycia's Week 7 code (as run) | Alycia |
| `prior_work/nazhan/` | Nazhan's Week 7 code (as run) | Nazhan |
| `analysis/dtd/` | DTD extra analysis (top-score spread) | Alycia |
| `.superset/`, `.githooks/`, `scripts/`, `.claude/`, `.gitignore`, this file | Repo setup and rules | Abby, after telling the team |

- Change files only in folders your person owns. If a task seems to need a change elsewhere, stop and tell the person what change is needed and who owns it.
- Every detector must write its output in the one format `shared/` defines, so one scoring script scores all three. Never invent a format of your own.

## NDA rules (the most important section)

The mentor allows coding agents to look at the pages, on one condition: the data must not be used for model training. So:

- You may open and inspect bank-statement pages, labels and masks on this laptop or on the Modal Volume **only after the person confirms "Help improve Claude" is off** (first steps, step 6). Look at as few pages as the task needs.
- Real pages live in a folder outside this repo and in the private Modal Volume `a4-data`. Never move, copy or link a page, mask or heat map into the repo folder.
- **The repo is public.** Nothing that identifies a page may be committed: no images of pages or crops, no heat maps, no real file names or doc IDs (a bank code, `_v` and a number, two underscores, then 10 hex characters), no bank names or codes, no statement text, no `sample_map.csv`. Only anonymous page IDs (`page_0001`) and numbers.
- Test order: 3 real pages at both sizes first (checks the output format and timing), then the full run. No fake pages.
- Figures in the repo show only numbers (charts). The write-up may show 1 to 2 anonymised synthetic sample pages, which are added in Overleaf by the team, never through the repo.
- `.githooks/pre-commit` blocks image, map and archive files outside `results/*/figures/`, large files, token-like strings, and anything that looks like a real file name. Never bypass it (`--no-verify` is forbidden).
- Never run `modal token` commands. The GPU token, the organisers' PDF and any `.env` file never go in the repo, a commit message, a printout or a chat.

## Git rules

- Everyone works on `main`. No branches or pull requests unless the person asks; the one exception is the local branch a Paradev workspace starts on (below).
- Before every push: `git pull --rebase` (or the Paradev steps below). Never force-push, rewrite history or run `git reset --hard` on shared work.
- Commit small and often. Start each message with the task ID, e.g. `T14: add DTD Modal job`.
- Stage only the files you changed for this task. Never `git add .` or `git add -A`.
- If a pull or rebase conflicts in a file you don't own, stop and tell the person.

### Who commits are credited to

- The author of every commit is the person whose computer made it, with their own git name and email. Never set Claude or any agent as author or committer, and never change anyone's git settings.
- **No `Co-Authored-By` line for any AI agent, in any commit or pull request.** `.claude/settings.json` turns Claude Code's line off; never add one by hand. The team credits AI coding help in the README and the write-up instead.

### Parallel agents (Paradev workspaces)

A Paradev workspace is its own git worktree on its own local branch. You're in one if `git branch --show-current` prints anything other than `main`.

- One task ID per workspace.
- Two workspaces running at once must not change the same file. Tell the person which files you will change.
- Only one workspace at a time may change `shared/`, this file, `.gitignore`, `.githooks/`, `.claude/` or `.superset/`.
- Never push the workspace branch or open a pull request from it. Deliver each small, working commit straight to `main`:
  1. `git fetch origin`
  2. `git rebase origin/main`
  3. `git push origin HEAD:main`
  If the push is rejected because `main` moved, repeat all three. Never force-push.
- If a rebase conflicts only in `docs/decisions.md`, keep both sides' lines in date order and continue. For any other conflict, run `git rebase --abort`, stop and tell the person.
- Never remove or prune worktrees. The person deletes workspaces in Paradev.

## GPU (Modal) rules

- The team shares 100 USD of credit and Modal has no automatic spending cut-off.
- Use `gpu="T4"` first. Every run starts with 3 real pages at both sizes: check the output matches `docs/PLAN.md` section 3, then tell the person the estimated GPU hours and cost for all pages from the seconds per page and Modal's current price list for that GPU. Wait for a yes before the rest runs.
- Run `shrink512` for all pages before `full`, so results exist early.
- Never save full heat maps; save only the compact `.npz` output in `docs/PLAN.md` section 3.
- Only one real run per detector at a time. Each detector gets its own Modal image (its own Python and package versions), so environments never clash.
- Save every run's settings (seed, sizes, weights, versions) next to its results.

## Honest results

- Never invent, estimate or hand-edit a number, score or table. If something was not run, say so.
- Every reported number comes from a results file in `results/`, with its 95% range and number of pages. Write "suggests", never "proves".
- Report weak and negative results. Report misses and false alarms separately.

## Figures

- **Every figure is drawn by matplotlib code**, run from a script saved in the repo next to the figure (e.g. `results/<detector>/figures/make_figures.py`). This covers every chart, table image and diagram used in the write-up, slides and video, including the pipeline diagram.
- **Never use AI image generators** (no generated images, illustrations, icons or "improved" versions of a chart from any AI tool, including Canva's AI features). Claude may write the matplotlib code; the picture itself always comes from running that code on our results files.
- Each figure is built only from a results file in `results/`, so anyone can rerun the script and get the same picture. Never hand-edit a figure.
- Write-up rules for figures: axis labels with units, a legend where there is more than one series, a short title, readable in greyscale (vary marker shape or line style, not only colour). Save as PNG at 300 dpi, plus PDF for Overleaf.
- The sample-page figure (1 to 2 anonymised synthetic pages, T46) is also drawn with matplotlib, but its script and output stay off the repo.

## Python rules

- Read and write every text file as UTF-8 and say so: `open(path, encoding="utf-8")`. Alycia is on Windows; Abby and Nazhan on Macs.
- Build paths with `pathlib`. No backslashes, no absolute paths to anyone's laptop.
- Set a seed for anything random, and record it.
- Give long scripts a small test mode (3 pages) and run it first.
- Add a new package with its version to the requirements file of the folder that needs it.

## How to explain your work

The team directs code through their agents and checks it by reading. None of them writes code from scratch.

- Put a short plain-language `#` comment above each line or small block of code, saying what it does.
- When you finish a task, say in plain words what changed, which files, and how the person can check it works.
- Record every method choice and every fix that could change a result in `docs/decisions.md`, as the "Decisions log" section below says.
- If something is unclear, ask one question before building. Don't guess.
- Never say a task is done unless you ran it and saw it work.
- Remind the person to update the task on the command centre sheet when a task is finished or blocked.

## Decisions log (`docs/decisions.md`)

This file is the team's full record of how the experiments were actually run (the command centre's Decisions tab is the summary copied from it). The write-up's Methodology and Limitations sections, and the answers to judges' "why did you do it this way?", are written from it. Keep it complete, even when rushing.

- **What gets a line:**
  - every method choice: input sizes and tiling, pre-processing (OCR masks, re-saves, colour or format conversion), thresholds and cut-offs, how ties and edge cases are handled, seeds, which weights and versions;
  - every fix that could change a number (a bug in reading pages, boxes, scores or IDs);
  - every difference from `docs/PLAN.md`, however small;
  - every run problem that affects results: pages that failed or were skipped and why, reruns, timeouts, out-of-memory, a changed GPU.
  Tidy-ups that can't change a number (renames, comments, formatting) need no line.
- **Format:** one dated line: what was chosen, why, and what it was chosen over. Add `Limitation:` when the choice limits what the results can claim.
  `- 2026-11-19 (T14, Alycia): DTD reads 512×512 tiles with no overlap, because 512 is DTD's input size. Chosen over overlapping tiles, which would double the GPU time. Limitation: an edit across a tile border is split between two tiles.`
- **When:** in the same commit as the code it describes. Never "later".
- **No results here.** Never write scores or other findings in this file; results live in `results/`.
- **Changing the plan:** `docs/PLAN.md` is frozen. If a choice would differ from it, stop and ask the person before building. If they agree, the decisions line records it, starting with `PLAN CHANGE:`.
- **End of every task:** list the decision lines you added in your summary, so the person can copy the method choices into the command centre's Decisions tab (you can't edit the sheet). If there are none, say so and confirm that no method choices were made.

## Which model

The team is on Claude Pro plans. Use Sonnet for writing and fixing code, Haiku for routine steps (starting a run, reading a table, small edits). Avoid Opus. If the person is on Opus for a routine step, say so once.
