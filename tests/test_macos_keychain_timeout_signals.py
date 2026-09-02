"""How a timed-out ``security`` spawn is stopped — SIGTERM only, never SIGKILL.

Why this file exists: ``subprocess.run(timeout=...)`` sends **SIGKILL**. When the
``security`` child is blocked on a SecurityAgent consent dialog that nobody can
answer (a shielded screen, or a headless launchd job), SIGKILL makes it vanish
with an outstanding XPC query still registered. ``securityd`` then tears that
query down, calls ``pthread_mutex_destroy`` on a still-held mutex, gets ``EBUSY``,
and the uncaught ``Security::UnixError`` reaches ``std::terminate`` -> ``abort()``.

The daemon dying is what actually hurts: the login keychain's master key lives
only in ``securityd``'s memory, so the respawn comes up **locked** and every app
on the machine re-prompts for the keychain password. Observed on 2026-08-25/26 as
16 ``securityd`` SIGABRT reports, with prompt-to-abort intervals of 4.877s and
4.970s against this module's 5.0s deadline; a prompt answered in 4.15s produced
no abort. Recurred 2026-09-02: SIGTERM-then-SIGKILL after 2s still SIGKILLed a
``security find-generic-password`` parked on the unlock dialog, aborted
``securityd`` twice at 14:42, and locked iMessage / Claude Code.

SIGTERM lets ``security`` unwind its own SecurityAgent query before exiting.
If it does not exit in the grace period, SIGKILL is still the bug — leave the
child. These tests assert SIGKILL is never sent.
"""

from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import patch

import pytest

from claude_swap import macos_keychain

pytestmark = pytest.mark.no_keychain_fake


@pytest.fixture(autouse=True)
def _isolate_keychain_log(tmp_path, monkeypatch):
    """Timeout logging must never touch the real ~/.claude state dir in tests."""
    monkeypatch.setenv("CLAUDE_SWAP_KEYCHAIN_LOG", str(tmp_path / "kc.jsonl"))


class _FakeProc:
    """``Popen`` stand-in that hangs until signalled, recording what it got.

    ``dies_on_term=False`` models a child wedged so hard SIGTERM doesn't land —
    the case that used to escalate to SIGKILL and abort securityd.
    """

    def __init__(self, *, dies_on_term: bool = True):
        self.signals: list[str] = []
        self.returncode: int | None = None
        self.pid: int | None = None
        self.stdin = self.stdout = self.stderr = None
        self._dies_on_term = dies_on_term

    def communicate(self, input=None, timeout=None):  # noqa: A002 - Popen's name
        if "kill" in self.signals:
            self.returncode = -9
            return ("", "")
        if "terminate" in self.signals and self._dies_on_term:
            self.returncode = -15
            return ("", "")
        raise subprocess.TimeoutExpired(cmd="security", timeout=timeout or 0)

    def terminate(self):
        self.signals.append("terminate")

    def kill(self):
        self.signals.append("kill")

    def poll(self):
        return self.returncode


# ---------------------------------------------------------------------------
# signal order
# ---------------------------------------------------------------------------


def test_timeout_sends_sigterm_and_stops_there_when_the_child_exits():
    proc = _FakeProc(dies_on_term=True)
    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=proc):
        with pytest.raises(subprocess.TimeoutExpired):
            macos_keychain._run_security(
                [macos_keychain._SECURITY, "find-generic-password"], timeout=0.01
            )
    # SIGKILL is what orphans securityd's query — a child that answers SIGTERM
    # must never be escalated to.
    assert proc.signals == ["terminate"]


def test_timeout_never_sends_sigkill_even_if_sigterm_is_ignored():
    proc = _FakeProc(dies_on_term=False)
    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=proc):
        with pytest.raises(subprocess.TimeoutExpired):
            macos_keychain._run_security(
                [macos_keychain._SECURITY, "find-generic-password"], timeout=0.01
            )
    # A SIGTERM-deaf security(1) parked on SecurityAgent is the crash:
    # SIGKILL orphans the query and securityd aborts. Leave the child.
    assert proc.signals == ["terminate"]


def test_timeout_still_surfaces_as_keychain_error_through_the_wrappers():
    # The public contract is unchanged: a wedged Keychain is a KeychainError,
    # never a hang and never a bare TimeoutExpired escaping the module.
    with patch(
        "claude_swap.macos_keychain.subprocess.Popen",
        side_effect=lambda *a, **k: _FakeProc(dies_on_term=True),
    ):
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain.get_password("svc", "acct")
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain.set_password("svc", "acct", "secret")
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain.delete_password("svc", "acct")
        # item_exists is deliberately non-raising.
        assert macos_keychain.item_exists("svc", "acct") is False


# ---------------------------------------------------------------------------
# the real thing — a live child process, no mocks
# ---------------------------------------------------------------------------


def test_real_child_receives_sigterm_not_sigkill(tmp_path):
    marker = tmp_path / "signal.txt"
    child = (
        "import signal, sys, time\n"
        f"open({str(marker)!r}, 'w').write('running')\n"
        "def handler(signum, frame):\n"
        f"    open({str(marker)!r}, 'w').write('SIGTERM')\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, handler)\n"
        "time.sleep(30)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        macos_keychain._run_security([sys.executable, "-c", child], timeout=0.75)
    # A SIGKILLed child cannot write this; only a handled SIGTERM can.
    assert marker.read_text() == "SIGTERM"


def test_real_child_ignoring_sigterm_is_not_sigkilled():
    child = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(30)\n"
    )
    real_popen = subprocess.Popen
    holder: dict[str, subprocess.Popen[str]] = {}

    def spy(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        holder["proc"] = proc
        return proc

    with patch.object(macos_keychain, "_TERM_GRACE", 0.4), patch(
        "claude_swap.macos_keychain.subprocess.Popen", side_effect=spy
    ):
        with pytest.raises(subprocess.TimeoutExpired):
            macos_keychain._run_security([sys.executable, "-c", child], timeout=0.4)
    proc = holder["proc"]
    try:
        # Still running: we raised TimeoutExpired without SIGKILL.
        assert proc.poll() is None
    finally:
        proc.kill()
        proc.wait(timeout=2)


# ---------------------------------------------------------------------------
# timeout breadcrumb (no secrets) — next incident's first read
# ---------------------------------------------------------------------------


def test_sanitize_security_argv_redacts_hex_payloads():
    argv = [
        "/usr/bin/security",
        "add-generic-password",
        "-U",
        "-a",
        "byeager",
        "-s",
        "Claude Code-credentials",
        "-X",
        "deadbeefcafesecret",
    ]
    out = macos_keychain._sanitize_security_argv(argv)
    assert "deadbeefcafesecret" not in out
    assert "-X" in out
    assert "<redacted>" in out
    assert "Claude Code-credentials" in out


def test_timeout_appends_jsonl_breadcrumb(tmp_path, monkeypatch):
    log_path = tmp_path / "kc.jsonl"
    monkeypatch.setenv("CLAUDE_SWAP_KEYCHAIN_LOG", str(log_path))
    monkeypatch.setenv("CLAUDE_SWAP_NO_KEYCHAIN", "1")
    proc = _FakeProc(dies_on_term=False)
    proc.pid = 4242
    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=proc):
        with pytest.raises(subprocess.TimeoutExpired):
            macos_keychain._run_security(
                [
                    macos_keychain._SECURITY,
                    "find-generic-password",
                    "-a",
                    "byeager",
                    "-w",
                    "-s",
                    "Claude Code-credentials",
                ],
                timeout=0.01,
            )
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["event"] == "security_timeout"
    assert row["child_alive_after_term"] is True
    assert row["no_keychain"] is True
    assert row["child_pid"] == 4242
    dumped = json.dumps(row)
    assert "sk-ant" not in dumped
    assert "-X" not in dumped or "<redacted>" in dumped
    assert "find-generic-password" in dumped


def test_timeout_log_failure_does_not_break_the_timeout_path(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_SWAP_KEYCHAIN_LOG", str(tmp_path / "nope" / "kc.jsonl"))
    with patch(
        "claude_swap.macos_keychain._keychain_log_path",
        side_effect=OSError("disk full"),
    ):
        proc = _FakeProc(dies_on_term=True)
        with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=proc):
            with pytest.raises(subprocess.TimeoutExpired):
                macos_keychain._run_security(
                    [macos_keychain._SECURITY, "find-generic-password"], timeout=0.01
                )
    assert proc.signals == ["terminate"]


# ---------------------------------------------------------------------------
# the deadline itself
# ---------------------------------------------------------------------------


def test_timeout_defaults_to_the_module_constant(monkeypatch):
    monkeypatch.delenv(macos_keychain._TIMEOUT_ENV, raising=False)
    assert macos_keychain._timeout() == macos_keychain._TIMEOUT


def test_timeout_honours_the_env_override(monkeypatch):
    # A host that can actually draw a dialog wants longer than 5s, so the human
    # gets a chance to answer before anything is killed at all.
    monkeypatch.setenv(macos_keychain._TIMEOUT_ENV, "20")
    assert macos_keychain._timeout() == 20.0


@pytest.mark.parametrize("bad", ["abc", "", "-1", "0", "nan"])
def test_timeout_ignores_a_garbage_override(monkeypatch, bad):
    # Never let a typo'd env var mean "no deadline" (a hang) or "kill instantly".
    monkeypatch.setenv(macos_keychain._TIMEOUT_ENV, bad)
    assert macos_keychain._timeout() == macos_keychain._TIMEOUT
