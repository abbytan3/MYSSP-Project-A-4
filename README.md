# Can Off-the-Shelf Tampering Detectors Find AI Edits in Bank Statements?

**MYSSP 2026, project A4** · Abby Tan, Alycia, Nazhan, with mentor Jia Yu Lim

> Work in progress. Results, figures and the final README are added before submission on 20 Nov 2026.

## What this project does

We test whether pre-trained document-tampering detectors (ADCD-Net, DTD and DAF), used exactly as released, can tell AI-edited bank statements from untouched ones and point to the edited field. AI edits (AnyText, UDiffText, TextCtrl) are compared with handcrafted edits (retyped text, copy-move, PDF edits) on real Malaysian bank statements, and with what human reviewers saw by eye. The full plan is in [`docs/PLAN.md`](docs/PLAN.md).

## Data

The bank statements come from an industry data partner under a non-disclosure agreement and were anonymised before we received them. No page, crop, file name or statement text is in this repository. Only score tables with anonymous page IDs and charts made from them are published here.

## Repository layout

| Folder | Contents |
|---|---|
| `shared/` | Sampling, upload, the common output format and the scoring script |
| `detectors/` | One folder per detector, with its cloud GPU job |
| `analysis/` | Human-rating analysis and the Week 7 ablation studies |
| `results/` | Score tables and charts per detector, and combined |
| `docs/` | Plan, setup guide, decisions log |

## Running the code

See [`docs/SETUP.md`](docs/SETUP.md) to install the tools. Instructions for each experiment are added with its code.

## Credits

Code in this repository was written with the help of AI coding assistants (Claude Code), directed and checked by the team. Detector code and weights belong to their authors; see the write-up for full references. Full acknowledgements are in the write-up.
