# Setup: GPU and coding space

Do this once, before your first task. It takes about 30 to 45 minutes. When you finish, mark task T06 and T07 Done on the command centre sheet.

**Three rules before you start**

- **Switch off "Help improve Claude"** at claude.ai → Settings → Privacy, before your coding agent touches any bank-statement page. Our mentor allows agents to see the pages only if the data isn't used for model training. (Task T44.)
- The Modal token in the organisers' PDF is private. Type it only into your own terminal (step 3). Never paste it into Claude, this repo, the sheet, a chat or a screenshot.
- The bank statements are under NDA. Never put a real page, a real file name or a bank name inside this repo folder. You won't need the pages on your laptop: Abby uploads the sample once to the team's private Modal storage (`a4-data`), and your detector job reads it there.

## What each step gives you

```
1 Git + GitHub      → you can download the code and save your work under your own name
2 Python 3.10+      → your computer can run Modal
3 Modal             → your computer can send jobs to the cloud GPU
4 Claude Code       → your coding agent
5 The repo          → the shared code and rules (AGENTS.md)
6 Paradev (Mac)     → several Claude Code sessions side by side, one per task
7 Check             → everything works before a real task
```

## 1. Git and your GitHub identity

**Mac:** Git comes with the Xcode command line tools. **Windows:** install [Git for Windows](https://git-scm.com/downloads/win). Claude Code also uses it.

Then, in a terminal:

```bash
# set the name that appears on your commits (your GitHub name)
git config --global user.name "Your GitHub name"
# set the email on your commits (one marked verified at github.com → Settings → Emails)
git config --global user.email "your-verified-email@example.com"
# check both are right
git config user.name
git config user.email
```

Every commit in this repo is authored by you, under these details. Claude is never listed as a co-author (the repo's `.claude/settings.json` turns that line off).

## 2. Python 3.10 or newer

Modal needs Python 3.10 to 3.14.

```bash
# check your Python version (Mac: python3 --version; Windows: python --version)
python3 --version
```

- **Mac:** the built-in Python may be 3.9, which is too old. Install a newer one with `brew install python@3.12`, or from [python.org](https://www.python.org/downloads/).
- **Windows:** install from [python.org](https://www.python.org/downloads/) and tick **Add python.exe to PATH** during install.

## 3. Modal (the cloud GPU)

Open the organisers' PDF, **"modal_gpu_quick_start_team3.pdf"**. Then, in your own terminal (not inside Claude Code):

```bash
# install or update the Modal tool
python3 -m pip install --upgrade modal
# log in with the team token: copy this one line from the PDF, section 1
modal token set --token-id <from the PDF> --token-secret <from the PDF>
# check it shows the MRG workspace
modal token info
```

On Windows, use `python` instead of `python3`.

Run the GPU check from the PDF, section 2: save its `gpu_check.py` in any folder **outside** the repo and run:

```bash
# start a short job on a T4 GPU; success = an NVIDIA table is printed
modal run gpu_check.py
```

**Costs:** the team shares 100 USD of GPU credit and nothing stops spending automatically. Always start on a T4, and run 3 real pages first to get the time per page and the cost before the full run.

## 4. Claude Code

```bash
# Mac (or Linux)
curl -fsSL https://claude.ai/install.sh | bash
```

```powershell
# Windows PowerShell (Alycia)
irm https://claude.ai/install.ps1 | iex
```

Open a **new** terminal window, then:

```bash
# check it installed; prints a version number
claude --version
```

The first time you run `claude`, it asks you to log in with your Claude Pro account. More help: [Claude Code quickstart](https://code.claude.com/docs/en/quickstart).

**Which model** (we are on Pro plans and can't wait for limits to reset):

| Work | Model | How |
|---|---|---|
| Writing or fixing code, careful reasoning | Sonnet 5.5 | type `/model` in Claude Code and pick Sonnet |
| Starting a run, reading a results table, small edits, updating the sheet | Haiku 5.5 | `/model`, pick Haiku |
| — | Avoid Opus | it uses up the Pro limit fastest |

## 5. The repo

Abby invites you to `abbytan3/MYSSP-Project-A-4` on GitHub. Accept the invite, then:

```bash
# go to the folder where you keep projects (example: Documents)
cd ~/Documents
# download the repo
git clone https://github.com/abbytan3/MYSSP-Project-A-4.git
# go into it
cd MYSSP-Project-A-4
# turn on the NDA check that runs before every commit (once per clone)
git config core.hooksPath .githooks
```

Keep the bank-statement data **outside** this folder, always.

## 6a. Mac: Paradev (Abby, Nazhan)

Paradev (formerly Superset) runs several Claude Code sessions side by side, each in its own copy of the repo (a "workspace"), so two tasks never overwrite each other. It runs on macOS only.

1. Install it from the Paradev website (download the Mac app; Apple Silicon or Intel build), and open it.
2. Add the `MYSSP-Project-A-4` folder as a project.
3. Create one workspace per task. The repo's `.superset/setup.sh` runs by itself when a workspace is created.
4. In the workspace, start Claude Code and send the first message (step 8).

## 6b. Windows: Claude Code directly (Alycia)

Paradev doesn't run on Windows, and you only need one task at a time.

1. Open PowerShell (or VS Code's terminal) in the `MYSSP-Project-A-4` folder.
2. Run `git pull` to get the latest code.
3. Run `claude` and send the first message (step 8).

## 7. Check before your first task

| Check | Command | Good result |
|---|---|---|
| Git name and email | `git config user.name` and `git config user.email` | Yours |
| Modal login | `modal token info` | The MRG workspace |
| GPU | `modal run gpu_check.py` | An NVIDIA table |
| Claude Code | `claude --version` | A version number |
| Repo up to date | `git pull` | "Already up to date" or new files |

Mark T06 and T07 Done on the command centre sheet.

## 8. Every task, from now on

Start each task with one message to Claude Code:

```text
I'm <Abby / Alycia / Nazhan>, working on task T<number>. Read AGENTS.md first and summarise the rules, then read docs/PLAN.md and my task.
```

Claude then checks your git details and that you're up to date, lists the files it will change, tests on 3 real pages, tells you the cost, and tells you the one command to start the real run yourself. When the task is done: check the result, commit, push, update the sheet.

## If something goes wrong

- `modal: command not found` → use `python3 -m modal ...` (Windows: `python -m modal ...`).
- `claude` not found → open a new terminal window; if it still fails, see [Troubleshoot installation](https://code.claude.com/docs/en/troubleshoot-install).
- Anything else → add a question on the sheet's **Questions** tab, or message the group.
