import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.script_smoke
@pytest.mark.parametrize(
    "module_name",
    [
        "scripts.train",
        "scripts.calibrate",
        "scripts.cross_validate",
        "scripts.interpolate_weights",
        "scripts.simulate_data",
    ],
)
def test_script_without_required_args_shows_usage(module_name: str):
    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    stderr = (result.stderr or "").lower()
    assert result.returncode != 0
    assert "usage:" in stderr
    assert "required" in stderr


@pytest.mark.script_smoke
@pytest.mark.parametrize(
    "module_name",
    [
        "scripts.train",
        "scripts.calibrate",
        "scripts.cross_validate",
        "scripts.interpolate_weights",
        "scripts.simulate_data",
    ],
)
def test_script_help_flag_exits_successfully(module_name: str):
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    output = ((result.stdout or "") + (result.stderr or "")).lower()
    assert result.returncode == 0
    assert "usage:" in output
