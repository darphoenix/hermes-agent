import os
from argparse import Namespace
from pathlib import Path

from hermes_cli.main import _apply_explicit_in_dir


def test_apply_explicit_in_dir_pins_process_and_terminal_cwd(tmp_path, monkeypatch):
    launch_dir = tmp_path / "launch"
    workspace = tmp_path / "workspace"
    launch_dir.mkdir()
    workspace.mkdir()
    monkeypatch.chdir(launch_dir)
    monkeypatch.setenv("TERMINAL_CWD", str(launch_dir))
    args = Namespace(in_dir=str(workspace), no_restore_cwd=False)

    in_dir, target_dir = _apply_explicit_in_dir(args)

    assert in_dir == str(workspace)
    assert target_dir == str(workspace)
    assert workspace == Path.cwd()
    assert str(workspace) == os.environ["TERMINAL_CWD"]
    assert args.no_restore_cwd is True
