"""NDA check run before every commit (called by .githooks/pre-commit).

Stops the commit if any staged file could leak confidential material into this public repo:
an image, heat map, archive or document outside the chart-figure folders, a very large file,
something that looks like a Modal token, a real page file name or doc ID, the sample map,
or a word from the optional local .nda_terms list.
"""

# read command-line arguments and exit with a status code
import sys
# run git commands from Python
import subprocess
# match file names and token-like text
import re
# build file paths that work on Mac and Windows
from pathlib import Path

# file types that may hold a page image, a heat map, model weights, an archive or a document
BLOCKED_TYPES = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif",
    ".npy", ".npz", ".pt", ".pth", ".ckpt",
    ".zip", ".tar", ".gz", ".tgz", ".7z", ".pdf",
}
# the only folders where PNG chart figures (made from score tables) are allowed
FIGURE_FOLDER = re.compile(r"^(results/[^/]+/figures|docs/figures)/[^/]+\.png$")
# anything bigger than 5 MB is suspicious in a repo of code and tables
MAX_BYTES = 5 * 1024 * 1024
# Modal token IDs and secrets start with ak- and as-
TOKEN_PATTERN = re.compile(r"\b(ak|as)-[A-Za-z0-9]{16,}\b")
# secret files that must never be committed
BLOCKED_NAMES = {".env", ".modal.toml", ".nda_terms", "sample_map.csv"}
# real page file names and doc IDs: a bank code, _v and a number, two underscores, 10 hex characters
REAL_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])[a-z]{3,5}_v\d+__[0-9a-f]{10}(?![0-9a-f])")


# list the files staged for this commit (added, copied, modified or renamed)
def staged_files():
    # ask git for the staged file names, one per line
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    # turn the output into a list, skipping blank lines
    return [line for line in out.stdout.splitlines() if line.strip()]


# read the staged version of a file (what will actually be committed)
def staged_bytes(path):
    # git show :path prints the staged content
    out = subprocess.run(["git", "show", f":{path}"], capture_output=True, check=True)
    # return the raw bytes
    return out.stdout


# load the optional local list of blocked words (one per line), kept out of git
def local_terms():
    # the file sits at the repo root and is git-ignored
    terms_file = Path(".nda_terms")
    # no file means no extra words to check
    if not terms_file.exists():
        return []
    # read the words, ignoring blank lines and lines starting with #
    lines = terms_file.read_text(encoding="utf-8").splitlines()
    # keep them lower-case so the check ignores capital letters
    return [w.strip().lower() for w in lines if w.strip() and not w.startswith("#")]


# check every staged file and collect the problems
def main():
    # start with no problems
    problems = []
    # the words from .nda_terms, if any
    terms = local_terms()
    # go through each staged file
    for path in staged_files():
        # use forward slashes so the folder rule works on Windows too
        p = path.replace("\\", "/")
        # the file's name and its type in lower case
        name = Path(p).name.lower()
        suffix = Path(p).suffix.lower()
        # block secret files by name
        if name in BLOCKED_NAMES or name.startswith(".env"):
            problems.append(f"{p}: secret file, never commit it")
            continue
        # block images, maps, archives and documents, except PNG charts in figure folders
        if suffix in BLOCKED_TYPES and not FIGURE_FOLDER.match(p):
            problems.append(f"{p}: {suffix} files are only allowed as charts in results/<detector>/figures/")
            continue
        # read what will be committed
        data = staged_bytes(p)
        # block very large files
        if len(data) > MAX_BYTES:
            problems.append(f"{p}: larger than 5 MB")
            continue
        # images in figure folders are binary, so skip the text checks for them
        if suffix == ".png":
            continue
        # decode as text, replacing anything that isn't valid UTF-8
        text = data.decode("utf-8", errors="replace")
        # block anything that looks like a Modal token
        if TOKEN_PATTERN.search(text):
            problems.append(f"{p}: contains something that looks like a Modal token")
        # block real page file names or doc IDs (use anonymous IDs like page_0001 instead)
        if REAL_ID_PATTERN.search(text):
            problems.append(f"{p}: contains a real page file name or doc ID; use anonymous page IDs")
        # block any word from the local list
        lowered = text.lower()
        for term in terms:
            if term in lowered:
                problems.append(f"{p}: contains a blocked word from .nda_terms")
                break
    # if anything was found, print it and stop the commit
    if problems:
        print("NDA check failed. Nothing was committed. Fix these files, then commit again:")
        for item in problems:
            print("  - " + item)
        print("If a file was staged by mistake: git restore --staged <file>")
        sys.exit(1)
    # otherwise let the commit go ahead
    print("NDA check passed.")


# run the check when the file is executed
if __name__ == "__main__":
    main()
