# AGENTS.md: rules for every coding agent in this repo

Read this whole file before doing anything. It applies to every session, whoever runs it and whichever agent they use (Claude Code reads it through `CLAUDE.md`).

## What this project is

MYSSP 2026, project A-4: "Can Off-the-Shelf Tampering Detectors Find AI Edits in Bank Statements?" Team: Abby, Alycia, Nazhan. Mentor and co-author: Jia Yu.

We tested whether pre-trained document-tampering detectors (ADCD-Net, DTD, DAF) catch AI text edits (AnyText, UDiffText, TextCtrl) and handcrafted edits on real Malaysian bank statements. Results are counted per document and per edited location, not per pixel.

- Detectors ran on the Modal cloud GPU. Code was written and tested on the team's laptops.
- The bank statements are under NDA. They never enter this repo or anything public (see NDA rules).
- **`docs/PLAN.md` is the confirmed experiment plan.** Never edit it; if it seems wrong, stop and ask. What was actually built, and every difference from the plan, is in `docs/decisions.md`.
- Task IDs such as T14 refer to the Tasks tab of the team's command centre sheet. The person tells you the ID and what the task says.

If this file and another file disagree, stop and ask.

## Project status (updated 24 Nov 2026)

The study is finished. The write-up, video and slides were submitted on 20 Nov 2026, and Demo Day is on 28 Nov 2026. This public repo is the record behind the submission, linked from the project page.

What may still change, and only when the person asks:

- Fixes the organisers ask for after their review (T39), and documentation corrections (READMEs, `SUMMARY.md` files, comments).
- Committing a missing script or a results file that backs a number already in the write-up.
- Nothing else. **Do not change any file in `results/` or `analysis/ablation/results/`, the scoring code in `shared/`, or a detector job, and do not run anything on Modal**, unless the person says the change is agreed with Abby. A changed number would no longer match the submitted write-up.

## First steps in every session

1. Ask who you're working with (Abby, Alycia or Nazhan) and the task ID, unless they've said.
2. Explain the task in 3 to 5 plain sentences before any work: what it does, why the project needs it, what comes before it and what uses its output next. Read `docs/PLAN.md` first if the task touches an experiment, so the explanation is right.
3. Summarise these rules in five lines so the person knows you read them.
4. Run `git config user.name` and `git config user.email`. If either is empty or not the person's own GitHub details, stop and ask them to fix it (`docs/SETUP.md`, step 1). Never change it yourself.
5. Run `git fetch origin` and `git status`. If this copy is behind `origin/main`, bring it up to date before changing anything (see Git rules).
6. Check the tools the task needs exist (`python3 --version` or `python --version`; `modal --version` only if the task needs Modal). Never run `modal token` commands: the token is private.
7. If the task touches bank-statement pages, ask the person to confirm "Help improve Claude" is switched off on their account (claude.ai → Settings → Privacy). Don't open any page until they confirm.
8. List the files this task will change and tell the person before you start.

## Who owns which folder

| Folder | Holds | Who changes it |
|---|---|---|
| `shared/` | Sampling, upload to Modal, output format, output check, scoring script | Abby |
| `detectors/adcd/` | ADCD-Net Modal job | Abby |
| `detectors/dtd/` | DTD Modal job | Alycia |
| `detectors/daf/` | DAF Modal job | Nazhan |
| `analysis/combined/`, `analysis/figures/make_figures.py` | Cross-detector tables and charts | Abby |
| `analysis/figures/make_figures_t16.py`, `analysis/figures/make_writeup_figures.py` | ADCD-Net charts; the write-up's Figures 2 to 5 | Abby |
| `analysis/figures/make_figures_t18.py` | DAF charts | Nazhan |
| `analysis/human/` | Stage 2 ratings: people vs detectors | Abby |
| `analysis/ablation/` | Week 7 ablation study, rescored | Abby |
| `analysis/tile_check/` | Page-size and tile-count checks | Abby |
| `analysis/dtd/` | DTD top-score spread | Alycia |
| `results/<detector>/` | Score tables and figures from that detector's runs | Whoever ran the run (adcd Abby, dtd Alycia, daf_anytext and daf_diffste Nazhan) |
| `results/combined/`, `results/human/`, `results/writeup/`, `results/README.md` | Cross-detector tables and charts; people vs detectors; the write-up's figures; the results guide | Abby |
| `docs/` | Setup guide, plan (frozen), decisions log | Anyone (`docs/PLAN.md`: nobody) |
| `prior_work/abby/` | Abby's Weeks 4 to 7 code (as run) | Abby |
| `prior_work/alycia/` | Alycia's Weeks 3 to 6 AnyText editing code (as run) | Alycia |
| `prior_work/nazhan/` | Nazhan's Week 7 DAF notebook (as run) | Nazhan |
| `README.md`, `.superset/`, `.githooks/`, `scripts/`, `.claude/`, `.gitignore`, this file | Repo front page, setup and rules | Abby, after telling the team |

- Change files only in folders your person owns. If a task seems to need a change elsewhere, stop and tell the person what change is needed and who owns it.
- Every detector writes its output in the one format `shared/output_format.py` defines, so one scoring script scores all three. Never invent a format of your own.

## NDA rules (the most important section)

The mentor allows coding agents to look at the pages, on one condition: the data must not be used for model training. So:

- You may open and inspect bank-statement pages, labels and masks on a team laptop or on the Modal Volume **only after the person confirms "Help improve Claude" is off** (first steps, step 7). Look at as few pages as the task needs.
- Real pages live in a folder outside this repo and in the private Modal Volume `a4-data`. Never move, copy or link a page, mask or heat map into the repo folder.
- **The repo is public.** Nothing that identifies a page may be committed: no images of pages or crops, no heat maps, no real file names or doc IDs (a bank code, `_v` and a number, two underscores, then 10 hex characters), no bank names or codes, no statement text, no `sample_map.csv`, no name of the shared Modal workspace. Only anonymous page IDs (`page_0001`) and numbers.
- Figures in the repo show only numbers (charts). The write-up may show 1 to 2 anonymised synthetic sample pages, which are added in Overleaf by the team, never through the repo.
- `.githooks/pre-commit` blocks image, map and archive files outside `results/*/figures/`, large files, token-like strings, and anything that looks like a real file name. Never bypass it (`--no-verify` is forbidden).
- Never run `modal token` commands. The GPU token, the organisers' PDF and any `.env` file never go in the repo, a commit message, a printout or a chat.
- Notebooks committed to `prior_work/` must have every output cleared and real paths replaced with `PATH_TO_*` placeholders.

## Git rules

- Everyone works on `main`. No branches or pull requests unless the person asks; the one exception is the local branch a Paradev workspace starts on (below).
- Before every push: `git pull --rebase` (or the Paradev steps below). Never force-push, rewrite history or run `git reset --hard` on shared work.
- Commit small and often. Start each message with the task ID, e.g. `T39: fix README link`. Documentation work with no task ID starts with `docs:`.
- Stage only the files you changed for this task. Never `git add .` or `git add -A`.
- If a pull or rebase conflicts in a file you don't own, stop and tell the person.

### Who commits are credited to

- The author of every commit is the person whose computer made it, with their own git name and email. Never set Claude or any agent as author or committer, and never change anyone's git settings.
- **No `Co-Authored-By` line for any AI agent, in any commit or pull request.** `.claude/settings.json` turns Claude Code's line off; never add one by hand. The team credits AI coding help in the README and the write-up instead.

### Parallel agents (Paradev workspaces)

A Paradev workspace is its own git worktree on its own local branch. You're in one if `git branch --show-current` prints anything other than `main`.

- One task ID per workspace.
- Two workspaces running at once must not change the same file. Tell the person which files you will change.
- Only one workspace at a time may change `shared/`, `README.md`, this file, `.gitignore`, `.githooks/`, `.claude/` or `.superset/`.
- Never push the workspace branch or open a pull request from it. Deliver each small, working commit straight to `main`:
  1. `git fetch origin`
  2. `git rebase origin/main`
  3. `git push origin HEAD:main`
  If the push is rejected because `main` moved, repeat all three. Never force-push.
- If a rebase conflicts only in `docs/decisions.md`, keep both sides' lines in date order and continue. For any other conflict, run `git rebase --abort`, stop and tell the person.
- Never remove or prune worktrees. The person deletes workspaces in Paradev.

## GPU (Modal) rules

All runs are finished. Do not start a Modal job unless the person says a rerun is agreed with Abby. If one is:

- The team shares 100 USD of credit and Modal has no automatic spending cut-off. Use `gpu="T4"`.
- Start with the 3 test pages at both sizes, check the output matches `docs/PLAN.md` section 3, tell the person the estimated GPU time and cost, and wait for a yes before the rest runs.
- Never save full heat maps; save only the compact `.npz` output. Save every run's settings next to its results.
- The Volume `a4-data` is to be deleted after the study (T52). Never copy a page out of it.

## Honest results

- Never invent, estimate or hand-edit a number, score or table. If something was not run, say so.
- Every reported number comes from a results file, with its 95% confidence interval and number of pages. Write "suggests", never "proves".
- Report weak and negative results. Report misses and false alarms separately.

## Wording rules (write-up, slides, project page and repo text)

- Write "95% confidence intervals" (or "95% CI"), not "95% ranges in brackets".
- Never describe an AUC as "chance". Say "close to 0.5", "the interval includes 0.5", or "no separation" (AUC 0.5).
- The Week 7 study is the "ablation study", never a "side study" or "controlled study".
- "Downscaled" in prose means `shrink512` in code and files; "Full" means `full`. Define it once per document.
- Read rule B only with its false-alarm rate, and location recall only with its false-region count.
- Pairwise group differences (editors, methods, capture conditions) were not computed: write "the intervals do not overlap, which suggests …", never "significantly different".
- The data partner is "AI Lens Sdn. Bhd., Kuala Lumpur". Never name the shared Modal workspace.

## Figures

- **Every figure is drawn by matplotlib code**, run from a script saved in the repo. This covers every chart, table image and diagram used in the write-up, slides and video.
- **Never use AI image generators** (no generated images, illustrations, icons or "improved" versions of a chart from any AI tool, including Canva's AI features). Claude may write the matplotlib code; the picture itself always comes from running that code on our results files.
- Each figure is built only from a results file, so anyone can rerun the script and get the same picture. Never hand-edit a figure.
- Figure rules: axis labels with units, a legend where there is more than one series, a short title, readable in greyscale (vary marker shape or line style, not only colour). Save as PNG at 300 dpi; PDFs stay off the repo (the NDA hook rejects them).
- The sample-page figure (1 to 2 anonymised synthetic pages) is also drawn with matplotlib, but its script and output stay off the repo.

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

This file is the team's full record of how the experiments were actually run (the command centre's Decisions tab is the summary copied from it). The write-up's Methodology and Limitations sections, and the answers to judges' "why did you do it this way?", are written from it.

- **What gets a line:** every method choice; every fix that could change a number; every difference from `docs/PLAN.md`, however small; every run problem that affects results. Tidy-ups that can't change a number (renames, comments, formatting, README wording) need no line.
- **Format:** one dated line: what was chosen, why, and what it was chosen over. Add `Limitation:` when the choice limits what the results can claim.
  `- 2026-11-19 (T14, Alycia): DTD reads 512×512 tiles with no overlap, because 512 is DTD's input size. Chosen over overlapping tiles, which would double the GPU time. Limitation: an edit across a tile border is split between two tiles.`
- **When:** in the same commit as the code it describes. Never "later".
- **No results here.** Never write scores or other findings in this file; results live in `results/`.
- **Changing the plan:** `docs/PLAN.md` is frozen. If they agree to a difference, the decisions line records it, starting with `PLAN CHANGE:`.
- **End of every task:** list the decision lines you added in your summary, so the person can copy the method choices into the command centre's Decisions tab (you can't edit the sheet). If there are none, say so.

## Which model

The team is on Claude Pro plans. Use Sonnet for writing and fixing code, Haiku for routine steps (reading a table, small edits). Avoid Opus. If the person is on Opus for a routine step, say so once.
