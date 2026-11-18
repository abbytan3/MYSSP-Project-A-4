# Decisions

One dated line per fix or decision: `- YYYY-MM-DD (task ID, name): what was decided and why.`

- 2026-11-17 (setup, Abby): Repo created from the planning-chat drafts. Rules for every coding agent are in AGENTS.md; CLAUDE.md points to it.
- 2026-11-17 (setup, Abby): Each member runs and analyses the detector they studied: Abby ADCD-Net, Alycia DTD, Nazhan DAF, all on one shared sample, scored by one shared script.
- 2026-11-17 (setup, Abby): The repo is public. Real bank-statement pages, their file names and heat maps never enter it; .gitignore and the pre-commit NDA check (scripts/check_nda.py) enforce this.
- 2026-11-17 (setup, Abby): Commits are authored by the person who makes them. No AI co-author line (.claude/settings.json turns Claude Code's off); AI coding help is credited in the README and the write-up.
- 2026-11-17 (setup, Abby): Detectors run on the Modal cloud GPU (organisers' 100 USD team credit). T4 first; timing on 20 fake pages before any real run.
- 2026-11-18 (plan, Abby): Mentor confirmed the plan (docs/PLAN.md): new title and three RQs; real statements only; all 506 AI-edited pages that passed stage 1 + 500 handcrafted + their originals; capture condition replaces our PNG/JPEG copies; pages with 2 edits kept; Week 7 study kept as ablation studies.
- 2026-11-18 (plan, Abby): AI editors are AnyText, UDiffText and TextCtrl (the mentor's dataset has no DiffSTE). DiffSTE stays in the Week 7 ablation.
- 2026-11-18 (plan, Abby): A page passes stage 1 if its reviewer's latest decision is "keep" (one reviewer per document).
- 2026-11-18 (plan, Abby): Coding agents may look at pages once their person has switched off "Help improve Claude" (mentor's condition). The public repo still never holds pages, file names or doc IDs.
- 2026-11-18 (plan, Abby): Rule B's cut-off is chosen by 2-fold cross-fitting over documents, since all kept AI pages are in the test sample and none are left to set aside.
- 2026-11-18 (plan, Abby): Stage 2 uses Abby, Nazhan and Alycia (same 1,000 pages); the helper's file has no page names and is left out. Only analysis: people vs detectors; no rater-agreement statistic. Score 5 = looks genuine, 1 = clearly edited.
- 2026-11-18 (plan, Abby): Rule B (cut-off by 2-fold cross-fitting) is the headline document rule; rule A (0.5, the authors' default) is reported alongside to show how it over-flags untouched pages.
- 2026-11-18 (process, Abby): Every agent explains the task in 3 to 5 sentences before starting (AGENTS.md, first steps).
- 2026-11-18 (process, Abby): Every figure (charts, tables, diagrams) is made with matplotlib from results files; no AI-generated images anywhere (AGENTS.md, Figures).
- 2026-11-18 (plan, Abby): No fake pages. Each run starts with 3 real pages at both sizes (format check, timing, cost estimate), then the full run; 512 for all detectors before full size.
- 2026-11-18 (plan, Abby): Full heat maps are not saved (over 100 GB). Each page keeps only its top score and the packed pixels at 0.5, which is all the document- and location-level scoring needs.
- 2026-11-18 (plan, Abby): DAF runs with two released weights, AnyText (editor in our set) and DiffSTE (diffusion editor not in our set); SRNet dropped. About 13,300 detector passes.
