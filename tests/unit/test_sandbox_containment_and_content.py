"""The sandbox trio: #800 (Python scripts never ran), #801 (raw-prefix containment), #802 (the path ran,
not the approved content).

Every gate is checked by a SIDE EFFECT (a marker file, a line of output) as well as the returned status:
a sandbox is proven by what did and did not happen.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from maxim.utils.sandbox_executor import ExecutionStatus, SandboxExecutor


@pytest.fixture
def executor(tmp_path: Path) -> SandboxExecutor:
    return SandboxExecutor(sandbox_dir=str(tmp_path / "sb"))


def _script(executor: SandboxExecutor, name: str, body: str) -> str:
    path = Path(executor.sandbox_dir) / "scripts" / name
    path.write_text(body)
    return str(path)


def _workspace(executor: SandboxExecutor) -> Path:
    return Path(executor.sandbox_dir) / "workspace"


# ── #800: a Python script runs, and cannot reach what the wrapper imported ──


def test_a_python_script_runs_and_writes_inside_the_sandbox(executor) -> None:
    script = _script(executor, "ok.py", "open('ran.txt', 'a').write('x')\nprint('hello')\n")
    result = executor.execute(script_path=script, require_approval=False)
    assert result.status == ExecutionStatus.SUCCESS, result.stderr
    assert (_workspace(executor) / "ran.txt").read_text() == "x"
    assert result.stdout.strip() == "hello"


@pytest.mark.parametrize("name", ["os", "sys", "real_open", "real_import", "_run", "_original_open"])
def test_the_script_cannot_see_the_wrappers_names(executor, name) -> None:
    """A naive fix (import os before the hook, exec in the wrapper's namespace) would hand these over."""
    script = _script(executor, "peek.py", f"{name}\n")
    result = executor.execute(script_path=script, require_approval=False)
    assert result.status == ExecutionStatus.FAILED
    assert "NameError" in result.stderr, result.stderr


def test_a_blocked_import_is_still_refused(executor) -> None:
    script = _script(executor, "net.py", "import socket\nopen('ran.txt', 'a').write('x')\n")
    result = executor.execute(script_path=script, require_approval=False)
    assert result.status == ExecutionStatus.FAILED
    assert "Import of 'socket' is not allowed in sandbox" in result.stderr  # the script's import, not the wrapper's
    assert not (_workspace(executor) / "ran.txt").exists()


def test_shell_arguments_and_script_name_survive_running_the_content(executor) -> None:
    """``bash -c <content> <path> <args>``: positional args land in $1.., and $0 is still the script."""
    script = _script(executor, "args.sh", 'echo "$(basename "$0") $1 $2"\n')
    result = executor.execute(script_path=script, args=["a", "b"], require_approval=False)
    assert result.status == ExecutionStatus.SUCCESS, result.stderr
    assert result.stdout.strip() == "args.sh a b"


# ── #801: containment is structural, after resolving symlinks ──


def test_a_sibling_directory_sharing_the_prefix_is_outside(executor, tmp_path) -> None:
    sibling = tmp_path / "sb2"  # the sandbox is tmp_path / "sb"
    sibling.mkdir()
    marker = sibling / "escaped.txt"
    evil = sibling / "evil.sh"
    evil.write_text(f"printf x > {marker}\n")
    result = executor.execute(script_path=str(evil), require_approval=False)
    assert result.status == ExecutionStatus.PERMISSION_DENIED
    assert not marker.exists()


def test_a_symlink_pointing_out_of_the_sandbox_is_outside(executor, tmp_path) -> None:
    outside = tmp_path / "outside.sh"
    marker = tmp_path / "escaped.txt"
    outside.write_text(f"printf x > {marker}\n")
    link = Path(executor.sandbox_dir) / "scripts" / "link.sh"
    os.symlink(outside, link)
    result = executor.execute(script_path=str(link), require_approval=False)
    assert result.status == ExecutionStatus.PERMISSION_DENIED
    assert not marker.exists()


@pytest.mark.parametrize("as_bytes", [False, True])
def test_a_python_script_cannot_open_a_sibling_sharing_the_prefix(executor, tmp_path, as_bytes) -> None:
    """Also as a BYTES path: str() of it is the literal "b'/...'", a relative name that resolves inside."""
    sibling = tmp_path / "sb2"
    sibling.mkdir()
    target = sibling / "secret.txt"
    target.write_text("secret")
    literal = repr(str(target).encode()) if as_bytes else repr(str(target))
    script = _script(executor, "read.py", f"print(open({literal}).read())\n")
    result = executor.execute(script_path=script, require_approval=False)
    assert result.status == ExecutionStatus.FAILED
    assert "secret" not in result.stdout


# ── #802: what runs is the approved content, never a re-read of the path ──


def test_a_line_appended_during_a_run_never_executes(executor) -> None:
    """The issue's reproduction: approve a slow script, append a line mid-run."""
    script = _script(executor, "slow.sh", "sleep 1\n")
    executor.approval_callback = lambda *_: True

    def append_later() -> None:
        time.sleep(0.4)
        with open(script, "a") as fh:
            fh.write("echo INJECTED\n")

    writer = threading.Thread(target=append_later)
    writer.start()
    result = executor.execute(script_path=script, require_approval=True)
    writer.join()

    assert result.status == ExecutionStatus.SUCCESS, result.stderr
    assert "INJECTED" not in result.stdout


@pytest.mark.parametrize(
    ("name", "approved", "on_disk"),
    [
        ("v.py", "print('approved')\n", "print('INJECTED')\n"),
        ("v.sh", "echo approved\n", "echo INJECTED\n"),
    ],
)
def test_the_run_executes_the_verified_content_not_the_file(executor, name, approved, on_disk) -> None:
    """Deterministic form of the race: the executor is handed the verified content while the file on
    disk says something else. Only the verified content may run -- for both script types."""
    script = _script(executor, name, on_disk)
    run = executor._execute_python if name.endswith(".py") else executor._execute_shell
    result = run(script, approved, [], None, str(_workspace(executor)))
    assert result.status == ExecutionStatus.SUCCESS, result.stderr
    assert result.stdout.strip() == "approved"


def test_no_wrapper_file_is_written_into_the_sandbox(executor) -> None:
    """Not a red gate (the old code unlinked its wrapper after the run): it pins that none is written,
    since a wrapper in the sandbox is a file the sandbox itself could rewrite."""
    script = _script(executor, "ok.py", "print('hi')\n")
    before = set(os.listdir(executor.sandbox_dir))
    executor.execute(script_path=script, require_approval=False)
    assert set(os.listdir(executor.sandbox_dir)) == before


# ── review folds ──


def test_a_working_dir_outside_the_sandbox_is_refused(executor, tmp_path) -> None:
    script = _script(executor, "pwd.sh", "pwd > where.txt\n")
    result = executor.execute(script_path=script, working_dir=str(tmp_path), require_approval=False)
    assert result.status == ExecutionStatus.PERMISSION_DENIED
    assert not (tmp_path / "where.txt").exists()


@pytest.mark.parametrize("name", ["big.sh", "big.py"])
def test_a_script_too_large_to_pass_is_refused_before_anyone_is_asked(executor, name) -> None:
    from maxim.utils.sandbox_executor import MAX_SCRIPT_BYTES

    asked = []
    executor.approval_callback = lambda *a: asked.append(a) or True
    script = _script(executor, name, "#" * (MAX_SCRIPT_BYTES + 1) + "\n")
    result = executor.execute(script_path=script, require_approval=True)
    assert result.status == ExecutionStatus.BLOCKED
    assert "too large for the sandbox" in (result.error or "")
    assert asked == []


@pytest.mark.parametrize("name", ["edge.sh", "edge.py"])
def test_a_script_at_the_limit_runs_whatever_its_bytes(executor, name) -> None:
    """The content travels verbatim as its own argument, so its size is its bytes -- even a worst case
    for escaping (a non-printable byte a repr would quadruple) runs at the limit. (macOS has no
    per-argument cap, so only Linux CI can turn this red.)"""
    from maxim.utils.sandbox_executor import MAX_SCRIPT_BYTES

    last = "echo ok\n" if name.endswith(".sh") else "print('ok')\n"
    filler = (
        ("x=" + "'" + "\x01" * (MAX_SCRIPT_BYTES - 64) + "'\n")
        if name.endswith(".py")
        else ("#" + "\x01" * (MAX_SCRIPT_BYTES - 64) + "\n")
    )
    script = _script(executor, name, filler + last)
    assert len(Path(script).read_bytes()) <= MAX_SCRIPT_BYTES
    result = executor.execute(script_path=script, require_approval=False)
    assert result.status == ExecutionStatus.SUCCESS, (result.error, result.stderr[-300:])
    assert result.stdout.strip() == "ok"


def test_a_script_with_a_nul_byte_is_refused_before_anyone_is_asked(executor) -> None:
    asked = []
    executor.approval_callback = lambda *a: asked.append(a) or True
    script = _script(executor, "nul.sh", "echo a\x00b\n")
    result = executor.execute(script_path=script, require_approval=True)
    assert result.status == ExecutionStatus.BLOCKED
    assert asked == []


def test_the_confined_open_opens_the_path_it_checked(executor, tmp_path) -> None:
    """A path object that answers differently the second time it is asked must not get past the check."""
    target = tmp_path / "outside.txt"
    target.write_text("SECRET")
    script = _script(
        executor,
        "twice.py",
        "class P:\n"
        "    n = 0\n"
        "    def __fspath__(self):\n"
        "        P.n += 1\n"
        f"        return 'ok.txt' if P.n == 1 else {str(target)!r}\n"
        "open('ok.txt', 'w').write('inside')\n"
        "print(open(P()).read())\n",
    )
    result = executor.execute(script_path=script, require_approval=False)
    assert "SECRET" not in result.stdout
    assert result.stdout.strip() == "inside"


def test_the_confined_open_takes_paths_only(executor) -> None:
    script = _script(executor, "fd.py", "open(0).read()\n")
    result = executor.execute(script_path=script, require_approval=False)
    assert result.status == ExecutionStatus.FAILED
    assert "Only file paths may be opened" in result.stderr


def test_a_script_does_not_read_the_hosts_stdin(executor) -> None:
    """The host's fd 0 is pointed at a pipe holding a line for the run: an inherited stdin would read it."""
    script = _script(executor, "stdin.sh", 'read -r line; echo "got:${line}"\n')
    read_end, write_end = os.pipe()
    os.write(write_end, b"host-secret\n")
    os.close(write_end)
    saved = os.dup(0)
    os.dup2(read_end, 0)
    try:
        result = executor.execute(script_path=script, require_approval=False)
    finally:
        os.dup2(saved, 0)
        os.close(saved)
        os.close(read_end)
    assert result.stdout.strip() == "got:"
