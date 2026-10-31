"""No tracked folder holds more than 20 files; a bigger one needs a parent that names its domain.

Owner, 2026-10-31: « We should never have like more than 20 files in a folder. I think we should instead try to find correct
parents, correct parent folders. »
"""

import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath

LIMIT = 20


def test_no_tracked_folder_holds_more_than_20_files() -> None:
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=Path(__file__).parents[1], check=True, capture_output=True, text=True).stdout.split("\0")
    sizes = Counter(str(PurePosixPath(path).parent) for path in tracked if path)
    over = {folder: count for folder, count in sizes.items() if count > LIMIT}
    assert not over, "folders over 20 files (find each its domain sub-folders):\n" + "\n".join(f"{count:4} {folder}" for folder, count in sorted(over.items()))
