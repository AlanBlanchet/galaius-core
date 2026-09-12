"""Build acceptance for the standalone distribution."""

import os
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_wheel_contains_code_and_generated_schema_resources(tmp_path: Path) -> None:
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=ROOT,
        check=True,
        env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"},
    )
    wheel, = tmp_path.glob("interact_core-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert "interact_core/__init__.py" in names
        assert {
            "interact_core/schema/account-contracts.schema.json",
            "interact_core/schema/admin-contracts.schema.json",
            "interact_core/schema/prompt-contracts.schema.json",
            "interact_core/schema/workflow-contracts.schema.json",
        } <= names
        metadata_name, = (name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode()
        assert "Requires-Dist: pydantic" in metadata
        assert "Requires-Dist: interact" not in metadata
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import interact_core",
            str(wheel),
        ],
        cwd=tmp_path,
        check=True,
    )
