"""Refuse ``security`` spawn when the Keychain is already prompting.

A second ``/usr/bin/security`` while SecurityAgent is up (or while a previous
child is still parked on that dialog) is the prompt-storm: each spawn raises
another unlock sheet, the 5s deadline expires, and the leftover children keep
the dialog alive. The gate in ``_run_security`` must refuse *before* Popen.

Never SIGKILL ``/usr/bin/security`` in these tests. Never call it to ask
whether the dialog is up — ``ps`` only, and ``_ps_text`` / ``_dialog_busy``
are injectable so the live process table is never the oracle.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import macos_keychain
from claude_swap import session as session_mod

# Captured at import, before the autouse quiet-stub replaces `_ps_text`.
_REAL_PS_TEXT = macos_keychain._ps_text

pytestmark = pytest.mark.no_keychain_fake

_FIND = [macos_keychain._SECURITY, "find-generic-password", "-a", "acct", "-s", "svc"]


@pytest.fixture(autouse=True)
def _isolate_keychain_state(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_SWAP_KEYCHAIN_LOG", str(tmp_path / "kc.jsonl"))
    monkeypatch.setenv("CLAUDE_SWAP_KEYCHAIN_CIRCUIT", str(tmp_path / "circuit.json"))
    monkeypatch.delenv("CLAUDE_SWAP_NO_KEYCHAIN", raising=False)
    monkeypatch.setattr(macos_keychain, "_ps_text", lambda: "", raising=False)
    if hasattr(macos_keychain, "_left_alive"):
        macos_keychain._left_alive = None
    return tmp_path


class _DoneProc:
    """A ``security`` child that returns immediately with a chosen rc."""

    def __init__(self, returncode: int = 0):
        self.returncode = returncode
        self.pid = 111
        self.stdin = self.stdout = self.stderr = None
        self.signals: list[str] = []

    def communicate(self, input=None, timeout=None):  # noqa: A002
        return ("", "")

    def poll(self):
        return self.returncode

    def terminate(self):
        self.signals.append("terminate")

    def kill(self):
        self.signals.append("kill")


def _log_rows(tmp_path: Path) -> list[dict]:
    log = tmp_path / "kc.jsonl"
    if not log.exists():
        return []
    text = log.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines()]


def _circuit(tmp_path: Path) -> dict | None:
    path = tmp_path / "circuit.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_circuit(tmp_path: Path, *, open_: bool, reason: str = "dialog_busy") -> None:
    (tmp_path / "circuit.json").write_text(
        json.dumps({"open": open_, "reason": reason}),
        encoding="utf-8",
    )


def _must_not_spawn(*_a, **_k):
    raise AssertionError("must not spawn /usr/bin/security")


# ---------------------------------------------------------------------------
# CLAUDE_SWAP_NO_KEYCHAIN=1
# ---------------------------------------------------------------------------


def test_no_keychain_env_refuses_without_popen(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_SWAP_NO_KEYCHAIN", "1")
    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn) as popen:
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain._run_security(_FIND)
    popen.assert_not_called()
    rows = _log_rows(tmp_path)
    assert rows, "expected security_spawn_refused breadcrumb"
    assert rows[-1]["event"] == "security_spawn_refused"
    assert rows[-1]["reason"] == "no_keychain_env"


def test_session_delete_cannot_bypass_no_keychain_env(tmp_path, monkeypatch):
    """session.py never checked the env; the gate in this module must cover it."""
    monkeypatch.setenv("CLAUDE_SWAP_NO_KEYCHAIN", "1")
    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn) as popen:
        session_mod.delete_macos_keychain_entry(tmp_path)
    popen.assert_not_called()


def test_only_explicit_1_is_no_keychain_env(monkeypatch):
    monkeypatch.setenv("CLAUDE_SWAP_NO_KEYCHAIN", "true")
    proc = _DoneProc(0)
    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=proc):
        result = macos_keychain._run_security(_FIND)
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# sticky circuit
# ---------------------------------------------------------------------------


def test_open_circuit_refuses_without_popen(tmp_path):
    _write_circuit(tmp_path, open_=True, reason="dialog_busy")
    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn) as popen:
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain._run_security(_FIND)
    popen.assert_not_called()
    assert _log_rows(tmp_path)[-1]["reason"] == "circuit_open"
    assert _circuit(tmp_path)["open"] is True


def test_circuit_does_not_auto_close_when_security_agent_is_gone(tmp_path, monkeypatch):
    _write_circuit(tmp_path, open_=True, reason="dialog_busy")
    monkeypatch.setattr(macos_keychain, "_dialog_busy", lambda: False, raising=False)
    monkeypatch.setattr(macos_keychain, "_ps_text", lambda: "", raising=False)
    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn) as popen:
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain._run_security(_FIND)
    popen.assert_not_called()
    assert _circuit(tmp_path)["open"] is True


def test_circuit_does_not_auto_close_on_a_timer(tmp_path, monkeypatch):
    _write_circuit(tmp_path, open_=True, reason="timeout")
    monkeypatch.setattr("time.monotonic", lambda: 10**12)
    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn):
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain._run_security(_FIND)
    assert _circuit(tmp_path)["open"] is True


def test_explicit_reset_closes_circuit_and_allows_spawn(tmp_path):
    _write_circuit(tmp_path, open_=True, reason="dialog_busy")
    macos_keychain.reset_keychain_circuit()
    assert _circuit(tmp_path)["open"] is False
    proc = _DoneProc(0)
    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=proc) as popen:
        result = macos_keychain._run_security(_FIND)
    popen.assert_called_once()
    assert result.returncode == 0
    assert _circuit(tmp_path)["open"] is False


def test_rc0_closes_circuit(tmp_path, monkeypatch):
    # Gate sees closed; the on-disk flag is still open so a successful spawn
    # is what actually writes the close — recovery after an explicit reset
    # that raced, or a helper that only skipped the check.
    _write_circuit(tmp_path, open_=True, reason="stale")
    monkeypatch.setattr(macos_keychain, "_circuit_is_open", lambda: False, raising=False)
    proc = _DoneProc(0)
    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=proc):
        macos_keychain._run_security(_FIND)
    assert _circuit(tmp_path)["open"] is False


def test_rc44_closes_circuit(tmp_path, monkeypatch):
    _write_circuit(tmp_path, open_=True, reason="stale")
    monkeypatch.setattr(macos_keychain, "_circuit_is_open", lambda: False, raising=False)
    proc = _DoneProc(44)
    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=proc):
        macos_keychain._run_security(_FIND)
    assert _circuit(tmp_path)["open"] is False


def test_other_rc_does_not_close_or_open_circuit(tmp_path):
    _write_circuit(tmp_path, open_=False)
    proc = _DoneProc(51)
    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=proc):
        macos_keychain._run_security(_FIND)
    assert _circuit(tmp_path)["open"] is False


def test_timeout_opens_circuit(tmp_path, monkeypatch):
    monkeypatch.setattr(macos_keychain, "_dialog_busy", lambda: False, raising=False)

    class _Hang:
        def __init__(self):
            self.returncode = None
            self.pid = 9
            self.stdin = self.stdout = self.stderr = None
            self.signals: list[str] = []

        def communicate(self, input=None, timeout=None):  # noqa: A002
            if "terminate" in self.signals:
                self.returncode = -15
                return ("", "")
            raise subprocess.TimeoutExpired(cmd="security", timeout=timeout or 0)

        def terminate(self):
            self.signals.append("terminate")

        def kill(self):
            self.signals.append("kill")

        def poll(self):
            return self.returncode

    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=_Hang()):
        with pytest.raises(subprocess.TimeoutExpired):
            macos_keychain._run_security(_FIND, timeout=0.01)
    assert _circuit(tmp_path)["open"] is True
    assert "kill" not in (_circuit(tmp_path) or {})


def test_circuit_default_path_is_under_keychain_watch(monkeypatch):
    monkeypatch.delenv("CLAUDE_SWAP_KEYCHAIN_CIRCUIT", raising=False)
    path = macos_keychain._circuit_path()
    assert path == Path.home() / ".claude" / "state" / "keychain-watch" / "circuit.json"


# ---------------------------------------------------------------------------
# process table — SecurityAgent or parked /usr/bin/security
# ---------------------------------------------------------------------------


def test_security_agent_in_ps_refuses_without_popen(tmp_path, monkeypatch):
    monkeypatch.setattr(
        macos_keychain,
        "_ps_text",
        lambda: "  4321  1  00:12 /System/Library/Frameworks/Security.framework/"
        "Versions/A/Resources/SecurityAgent.app/Contents/MacOS/SecurityAgent",
        raising=False,
    )
    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn) as popen:
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain._run_security(_FIND)
    popen.assert_not_called()
    assert _log_rows(tmp_path)[-1]["reason"] == "dialog_busy"
    assert _circuit(tmp_path)["open"] is True


def test_parked_security_in_ps_refuses_without_popen(tmp_path, monkeypatch):
    monkeypatch.setattr(
        macos_keychain,
        "_ps_text",
        lambda: "  4242  88  00:07 /usr/bin/security find-generic-password -a byeager -s svc",
        raising=False,
    )
    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn) as popen:
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain._run_security(_FIND)
    popen.assert_not_called()
    assert _log_rows(tmp_path)[-1]["reason"] == "dialog_busy"


def test_dialog_busy_override_refuses_without_popen(tmp_path, monkeypatch):
    monkeypatch.setattr(macos_keychain, "_dialog_busy", lambda: True, raising=False)
    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn) as popen:
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain._run_security(_FIND)
    popen.assert_not_called()
    assert _log_rows(tmp_path)[-1]["reason"] == "dialog_busy"


def test_quiet_process_table_allows_spawn(monkeypatch):
    monkeypatch.setattr(macos_keychain, "_ps_text", lambda: "  1  0  01:00 /sbin/launchd", raising=False)
    proc = _DoneProc(0)
    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=proc) as popen:
        result = macos_keychain._run_security(_FIND)
    popen.assert_called_once()
    assert result.returncode == 0


def test_ps_text_invokes_ps_not_security():
    with patch("claude_swap.macos_keychain.subprocess.run") as run:
        run.return_value = subprocess.CompletedProcess(
            args=["ps"], returncode=0, stdout="", stderr=""
        )
        _REAL_PS_TEXT()
    argv = [str(a) for a in run.call_args.args[0]]
    assert argv[0] in ("ps", "/bin/ps")
    assert "-axo" in argv
    assert "pid=,ppid=,etime=,command=" in argv
    assert all("/usr/bin/security" not in a for a in argv)


def test_refuse_reason_prefers_no_keychain_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_SWAP_NO_KEYCHAIN", "1")
    _write_circuit(tmp_path, open_=True)
    monkeypatch.setattr(macos_keychain, "_dialog_busy", lambda: True, raising=False)
    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn):
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain._run_security(_FIND)
    assert _log_rows(tmp_path)[-1]["reason"] == "no_keychain_env"


# ---------------------------------------------------------------------------
# breadcrumb — no secrets
# ---------------------------------------------------------------------------


def test_spawn_refused_breadcrumb_redacts_hex_and_password(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_SWAP_NO_KEYCHAIN", "1")
    argv = [
        macos_keychain._SECURITY,
        "add-generic-password",
        "-a",
        "byeager",
        "-s",
        "Claude Code-credentials",
        "-X",
        "deadbeefcafesecret",
        "-p",
        "sk-ant-literal-secret",
    ]
    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn):
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain._run_security(argv)
    dumped = json.dumps(_log_rows(tmp_path)[-1])
    assert "deadbeefcafesecret" not in dumped
    assert "sk-ant-literal-secret" not in dumped
    assert "<redacted>" in dumped
    assert "add-generic-password" in dumped


# ---------------------------------------------------------------------------
# wrappers — refuse is a KeychainError; item_exists stays non-raising
# ---------------------------------------------------------------------------


def test_wrappers_refuse_without_spawn_and_item_exists_is_false(monkeypatch):
    monkeypatch.setenv("CLAUDE_SWAP_NO_KEYCHAIN", "1")
    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn) as popen:
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain.get_password("svc", "acct")
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain.set_password("svc", "acct", "secret")
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain.delete_password("svc", "acct")
        assert macos_keychain.item_exists("svc", "acct") is False
    popen.assert_not_called()
