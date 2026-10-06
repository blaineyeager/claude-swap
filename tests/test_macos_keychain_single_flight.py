"""One left-alive ``security`` child blocks the next spawn.

A timeout that cannot SIGTERM a child parked on SecurityAgent must leave that
child running (SIGKILL aborts securityd). The next ``_run_security`` in this
process must refuse rather than stack another dialog on top of it.

Never SIGKILL ``/usr/bin/security`` here. The leftover is a FakeProc.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from claude_swap import macos_keychain

pytestmark = pytest.mark.no_keychain_fake

_FIND = [macos_keychain._SECURITY, "find-generic-password"]


@pytest.fixture(autouse=True)
def _isolate_keychain_state(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_SWAP_KEYCHAIN_LOG", str(tmp_path / "kc.jsonl"))
    monkeypatch.setenv("CLAUDE_SWAP_KEYCHAIN_CIRCUIT", str(tmp_path / "circuit.json"))
    monkeypatch.delenv("CLAUDE_SWAP_NO_KEYCHAIN", raising=False)
    monkeypatch.setattr(macos_keychain, "_ps_text", lambda: "", raising=False)
    monkeypatch.setattr(
        macos_keychain, "_dialog_busy", lambda exclude_pid=None: False, raising=False
    )
    if hasattr(macos_keychain, "_left_alive"):
        macos_keychain._left_alive = None
    return tmp_path


class _FakeProc:
    def __init__(self, *, dies_on_term: bool = False):
        self.signals: list[str] = []
        self.returncode: int | None = None
        self.pid: int | None = 4242
        self.stdin = self.stdout = self.stderr = None
        self._dies_on_term = dies_on_term

    def communicate(self, input=None, timeout=None):  # noqa: A002
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


def _must_not_spawn(*_a, **_k):
    raise AssertionError("must not spawn /usr/bin/security")


def _close_circuit(tmp_path):
    (tmp_path / "circuit.json").write_text(
        json.dumps({"open": False}), encoding="utf-8"
    )


def _last_reason(tmp_path) -> str:
    lines = (tmp_path / "kc.jsonl").read_text(encoding="utf-8").strip().splitlines()
    return json.loads(lines[-1])["reason"]


def test_left_alive_child_blocks_the_next_spawn(tmp_path):
    leftover = _FakeProc(dies_on_term=False)
    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=leftover):
        with pytest.raises(subprocess.TimeoutExpired):
            macos_keychain._run_security(_FIND, timeout=0.01)
    assert "kill" not in leftover.signals
    # Timeout also opens the circuit; close it so the reason is single_flight.
    _close_circuit(tmp_path)

    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn) as popen:
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain._run_security(_FIND)
    popen.assert_not_called()
    assert _last_reason(tmp_path) == "single_flight"
    assert leftover.poll() is None
    assert "kill" not in leftover.signals


def test_dialog_timeout_records_child_for_single_flight(tmp_path, monkeypatch):
    # Gate must let the first spawn through; timeout then sees the dialog.
    calls = {"n": 0}

    def busy(exclude_pid=None) -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    monkeypatch.setattr(macos_keychain, "_dialog_busy", busy, raising=False)
    leftover = _FakeProc(dies_on_term=False)
    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=leftover):
        with pytest.raises(subprocess.TimeoutExpired):
            macos_keychain._run_security(_FIND, timeout=0.01)
    assert leftover.signals == []  # no SIGTERM, no SIGKILL
    _close_circuit(tmp_path)
    monkeypatch.setattr(
        macos_keychain, "_dialog_busy", lambda exclude_pid=None: False, raising=False
    )

    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn) as popen:
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain._run_security(_FIND)
    popen.assert_not_called()
    assert _last_reason(tmp_path) == "single_flight"


def test_exited_leftover_does_not_block_the_next_spawn(tmp_path):
    leftover = _FakeProc(dies_on_term=True)
    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=leftover):
        with pytest.raises(subprocess.TimeoutExpired):
            macos_keychain._run_security(_FIND, timeout=0.01)
    assert leftover.poll() is not None
    _close_circuit(tmp_path)

    class _Done:
        def __init__(self):
            self.returncode = 0
            self.pid = 7
            self.stdin = self.stdout = self.stderr = None

        def communicate(self, input=None, timeout=None):  # noqa: A002
            return ("", "")

        def poll(self):
            return 0

    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=_Done()) as popen:
        result = macos_keychain._run_security(_FIND)
    popen.assert_called_once()
    assert result.returncode == 0
