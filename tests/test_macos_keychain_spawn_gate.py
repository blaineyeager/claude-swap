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
import time
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
    monkeypatch.delenv("CLAUDE_SWAP_KEYCHAIN_CIRCUIT_TTL", raising=False)
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


def _stray_temps(tmp_path: Path) -> list[str]:
    """Names left behind by a circuit write other than the published file."""
    kept = {"circuit.json", "kc.jsonl"}
    return sorted(p.name for p in tmp_path.iterdir() if p.name not in kept)


def _write_circuit(
    tmp_path: Path,
    *,
    open_: bool,
    reason: str = "dialog_busy",
    ts: str | None = None,
) -> None:
    """Write a circuit file. ``ts`` defaults to now so an open circuit is fresh."""
    if ts is None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (tmp_path / "circuit.json").write_text(
        json.dumps({"open": open_, "reason": reason, "ts": ts}),
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


def test_no_keychain_env_refuse_does_not_open_sticky_circuit(tmp_path, monkeypatch):
    """CLAUDE_SWAP_NO_KEYCHAIN=1 is a process opt-out, not a durable host fault.

    Opening circuit.json here would keep refusing after the env is gone,
    until the TTL or ``cswap keychain-circuit reset``.
    """
    monkeypatch.setenv("CLAUDE_SWAP_NO_KEYCHAIN", "1")
    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn) as popen:
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain._run_security(_FIND)
    popen.assert_not_called()
    circuit = _circuit(tmp_path)
    assert circuit is None or circuit.get("open") is not True

    monkeypatch.delenv("CLAUDE_SWAP_NO_KEYCHAIN", raising=False)
    monkeypatch.setattr(macos_keychain, "_ps_text", lambda: "", raising=False)
    proc = _DoneProc(0)
    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=proc) as popen:
        result = macos_keychain._run_security(_FIND)
    popen.assert_called_once()
    assert result.returncode == 0


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
# spawn circuit
# ---------------------------------------------------------------------------


def test_open_circuit_refuses_without_popen(tmp_path):
    _write_circuit(tmp_path, open_=True, reason="dialog_busy")
    before = (tmp_path / "circuit.json").read_bytes()
    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn) as popen:
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain._run_security(_FIND)
    popen.assert_not_called()
    assert _log_rows(tmp_path)[-1]["reason"] == "circuit_open"
    # A circuit_open refuse must not rewrite the file (that restarts the TTL).
    assert (tmp_path / "circuit.json").read_bytes() == before
    assert _circuit(tmp_path)["open"] is True


def test_circuit_does_not_auto_close_when_security_agent_is_gone(tmp_path, monkeypatch):
    _write_circuit(tmp_path, open_=True, reason="dialog_busy")
    monkeypatch.setattr(
        macos_keychain, "_dialog_busy", lambda exclude_pid=None: False, raising=False
    )
    monkeypatch.setattr(macos_keychain, "_ps_text", lambda: "", raising=False)
    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn) as popen:
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain._run_security(_FIND)
    popen.assert_not_called()
    assert _circuit(tmp_path)["open"] is True


def test_fresh_circuit_still_refuses_and_stays_open(tmp_path):
    """A circuit opened just now (inside the TTL) still refuses and stays open."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_circuit(tmp_path, open_=True, reason="timeout", ts=ts)
    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn) as popen:
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain._run_security(_FIND)
    popen.assert_not_called()
    assert _circuit(tmp_path)["open"] is True


def test_expired_circuit_is_closed_and_allows_spawn(tmp_path, monkeypatch):
    """An open circuit older than the TTL is persisted closed and spawn proceeds."""
    monkeypatch.setenv("CLAUDE_SWAP_KEYCHAIN_CIRCUIT_TTL", "60")
    old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 2 * 60 * 60))
    _write_circuit(tmp_path, open_=True, reason="timeout", ts=old)

    # The gate itself persists the close; rc 0 would also close afterwards.
    assert macos_keychain._circuit_is_open() is False
    closed = _circuit(tmp_path)
    assert closed is not None
    assert closed["open"] is False
    assert closed["reason"] == "expired"

    proc = _DoneProc(0)
    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=proc) as popen:
        result = macos_keychain._run_security(_FIND)
    popen.assert_called_once()
    assert result.returncode == 0
    assert _circuit(tmp_path)["open"] is False


def test_circuit_ttl_env_override_honored(tmp_path, monkeypatch):
    """``CLAUDE_SWAP_KEYCHAIN_CIRCUIT_TTL`` changes the deadline; bad values do not."""
    assert macos_keychain._circuit_ttl() == macos_keychain._CIRCUIT_TTL_DEFAULT
    monkeypatch.setenv("CLAUDE_SWAP_KEYCHAIN_CIRCUIT_TTL", "60")
    assert macos_keychain._circuit_ttl() == 60.0
    monkeypatch.setenv("CLAUDE_SWAP_KEYCHAIN_CIRCUIT_TTL", "90.5")
    assert macos_keychain._circuit_ttl() == 90.5
    for bad in ("", "nope", "-1", "-0.1", "0", "0.0", "-0", "nan", "inf"):
        monkeypatch.setenv("CLAUDE_SWAP_KEYCHAIN_CIRCUIT_TTL", bad)
        assert macos_keychain._circuit_ttl() == macos_keychain._CIRCUIT_TTL_DEFAULT

    # Ten minutes old would expire at the 300s default. An override inside
    # the cap keeps it open. Values above the cap are clamped; see
    # test_circuit_ttl_is_capped.
    monkeypatch.setenv("CLAUDE_SWAP_KEYCHAIN_CIRCUIT_TTL", "3600")
    old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 10 * 60))
    _write_circuit(tmp_path, open_=True, reason="timeout", ts=old)
    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn) as popen:
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain._run_security(_FIND)
    popen.assert_not_called()
    assert _circuit(tmp_path)["open"] is True

    # The same age expires when the override is shorter than it.
    monkeypatch.setenv("CLAUDE_SWAP_KEYCHAIN_CIRCUIT_TTL", "60")
    _write_circuit(tmp_path, open_=True, reason="timeout", ts=old)
    proc = _DoneProc(0)
    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=proc) as popen:
        result = macos_keychain._run_security(_FIND)
    popen.assert_called_once()
    assert result.returncode == 0
    assert _circuit(tmp_path)["open"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"open": True, "reason": "dialog_busy"},
        {"open": True, "reason": "dialog_busy", "ts": "not-a-timestamp"},
        {"open": True, "reason": "dialog_busy", "ts": ""},
    ],
    ids=["missing-ts", "unparseable-ts", "empty-ts"],
)
def test_circuit_missing_or_bad_ts_is_treated_closed(tmp_path, payload):
    """A corrupt timestamp must not wedge spawns; the circuit is persisted closed."""
    (tmp_path / "circuit.json").write_text(json.dumps(payload), encoding="utf-8")
    assert macos_keychain._circuit_is_open() is False
    closed = _circuit(tmp_path)
    assert closed is not None and closed["open"] is False
    assert closed["reason"] == "expired"

    proc = _DoneProc(0)
    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=proc) as popen:
        result = macos_keychain._run_security(_FIND)
    popen.assert_called_once()
    assert result.returncode == 0
    assert _circuit(tmp_path)["open"] is False


def test_keychain_circuit_reset_command_closes_and_prints(tmp_path, capsys):
    import sys

    from claude_swap import cli

    _write_circuit(tmp_path, open_=True, reason="dialog_busy")
    assert _circuit(tmp_path)["open"] is True
    with patch.object(sys, "argv", ["cswap", "keychain-circuit", "reset"]):
        cli.main()
    assert "Keychain spawn circuit reset (closed)" in capsys.readouterr().out
    assert _circuit(tmp_path)["open"] is False


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
    monkeypatch.setattr(
        macos_keychain, "_dialog_busy", lambda exclude_pid=None: False, raising=False
    )

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


def test_circuit_expires_exactly_at_ttl(tmp_path, monkeypatch):
    """age == TTL is expired; one epsilon inside the window is still open."""
    ttl = 60.0
    epsilon = 1e-3
    monkeypatch.setenv("CLAUDE_SWAP_KEYCHAIN_CIRCUIT_TTL", "60")
    opened = 1_700_000_000
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(opened))
    assert macos_keychain._circuit_ts_epoch(ts) == opened
    _write_circuit(tmp_path, open_=True, reason="timeout", ts=ts)
    before = (tmp_path / "circuit.json").read_bytes()

    monkeypatch.setattr(macos_keychain.time, "time", lambda: opened + ttl - epsilon)
    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn) as popen:
        with pytest.raises(macos_keychain.KeychainError, match="circuit_open"):
            macos_keychain._run_security(_FIND)
    popen.assert_not_called()
    assert (tmp_path / "circuit.json").read_bytes() == before
    assert macos_keychain._circuit_is_open() is True

    monkeypatch.setattr(macos_keychain.time, "time", lambda: opened + ttl)
    assert macos_keychain._circuit_is_open() is False
    closed = _circuit(tmp_path)
    assert closed is not None
    assert closed["open"] is False
    assert closed["reason"] == "expired"

    proc = _DoneProc(0)
    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=proc) as popen:
        result = macos_keychain._run_security(_FIND)
    popen.assert_called_once()
    assert result.returncode == 0
    assert _circuit(tmp_path)["open"] is False


def test_circuit_ttl_zero_falls_back_to_default(tmp_path, monkeypatch):
    """``0`` / ``0.0`` / ``-0`` are not a positive TTL; they fall back to 300s.

    A literal zero would expire every open circuit immediately and disable
    the breaker. ``float("-0")`` is ``-0.0``, which must not slip past a
    ``< 0`` check.
    """
    for raw in ("0", "0.0", "-0", "+0", "-0.0"):
        monkeypatch.setenv("CLAUDE_SWAP_KEYCHAIN_CIRCUIT_TTL", raw)
        assert macos_keychain._circuit_ttl() == macos_keychain._CIRCUIT_TTL_DEFAULT

    # One minute old: expired if TTL were really 0, still inside the default.
    monkeypatch.setenv("CLAUDE_SWAP_KEYCHAIN_CIRCUIT_TTL", "0")
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 60))
    _write_circuit(tmp_path, open_=True, reason="timeout", ts=ts)
    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn) as popen:
        with pytest.raises(macos_keychain.KeychainError, match="circuit_open"):
            macos_keychain._run_security(_FIND)
    popen.assert_not_called()
    circuit = _circuit(tmp_path)
    assert circuit["open"] is True
    assert circuit["ts"] == ts


def test_circuit_ttl_is_capped(tmp_path, monkeypatch):
    """Overrides above ``_CIRCUIT_TTL_MAX`` clamp; they do not pin the breaker."""
    assert macos_keychain._CIRCUIT_TTL_MAX == 3600.0
    for raw in ("3600", "3600.0"):
        monkeypatch.setenv("CLAUDE_SWAP_KEYCHAIN_CIRCUIT_TTL", raw)
        assert macos_keychain._circuit_ttl() == 3600.0
    for raw in ("3600.1", "86400", "1e6"):
        monkeypatch.setenv("CLAUDE_SWAP_KEYCHAIN_CIRCUIT_TTL", raw)
        assert macos_keychain._circuit_ttl() == macos_keychain._CIRCUIT_TTL_MAX

    # Older than the cap, younger than the uncapped override: expired.
    monkeypatch.setenv("CLAUDE_SWAP_KEYCHAIN_CIRCUIT_TTL", "86400")
    stale = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - (3600 + 120))
    )
    _write_circuit(tmp_path, open_=True, reason="timeout", ts=stale)
    assert macos_keychain._circuit_is_open() is False
    assert _circuit(tmp_path)["reason"] == "expired"

    # Younger than the cap stays open even when the env asked for a day.
    fresh = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 60))
    _write_circuit(tmp_path, open_=True, reason="timeout", ts=fresh)
    with patch("claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn) as popen:
        with pytest.raises(macos_keychain.KeychainError, match="circuit_open"):
            macos_keychain._run_security(_FIND)
    popen.assert_not_called()
    assert _circuit(tmp_path)["open"] is True
    assert _circuit(tmp_path)["ts"] == fresh


def test_future_circuit_timestamp_beyond_skew_is_expired(tmp_path, monkeypatch):
    """A stamp further ahead than the skew allowance is a clock jump.

    Within the allowance (and exactly at it) the circuit stays open. One
    second past it is persisted closed so spawns are not pinned until wall
    time catches a stamp from before the step.
    """
    now = 1_700_000_000.0
    skew = macos_keychain._CIRCUIT_FUTURE_SKEW
    assert skew == 60.0
    monkeypatch.setattr(macos_keychain.time, "time", lambda: now)

    for ahead in (30, int(skew)):
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + ahead))
        _write_circuit(tmp_path, open_=True, reason="timeout", ts=ts)
        before = (tmp_path / "circuit.json").read_bytes()
        with patch(
            "claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn
        ) as popen:
            with pytest.raises(macos_keychain.KeychainError, match="circuit_open"):
                macos_keychain._run_security(_FIND)
        popen.assert_not_called()
        assert (tmp_path / "circuit.json").read_bytes() == before
        assert macos_keychain._circuit_is_open() is True

    beyond = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + skew + 1))
    _write_circuit(tmp_path, open_=True, reason="timeout", ts=beyond)
    assert macos_keychain._circuit_is_open() is False
    closed = _circuit(tmp_path)
    assert closed is not None
    assert closed["open"] is False
    assert closed["reason"] == "expired"

    proc = _DoneProc(0)
    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=proc) as popen:
        result = macos_keychain._run_security(_FIND)
    popen.assert_called_once()
    assert result.returncode == 0
    assert _circuit(tmp_path)["open"] is False


def test_circuit_open_refuse_does_not_restart_ttl(tmp_path, monkeypatch):
    """Polling once a minute inside the TTL must not refresh ``ts``.

    Each ``circuit_open`` refuse used to rewrite the circuit file, so the
    window restarted and a poller never recovered. The original stamp is
    kept, and a spawn goes through at ``original ts + TTL``.
    """
    ttl = 300.0
    monkeypatch.setenv("CLAUDE_SWAP_KEYCHAIN_CIRCUIT_TTL", "300")
    assert macos_keychain._circuit_ttl() == ttl
    opened = 1_700_000_000
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(opened))
    _write_circuit(tmp_path, open_=True, reason="dialog_busy", ts=ts)
    original = (tmp_path / "circuit.json").read_bytes()

    clock = {"now": float(opened)}
    monkeypatch.setattr(macos_keychain.time, "time", lambda: clock["now"])

    for step in range(0, 5):  # 0, 60, 120, 180, 240 — all strictly inside 300
        clock["now"] = opened + step * 60.0
        assert clock["now"] < opened + ttl
        with patch(
            "claude_swap.macos_keychain.subprocess.Popen", side_effect=_must_not_spawn
        ) as popen:
            with pytest.raises(macos_keychain.KeychainError, match="circuit_open"):
                macos_keychain._run_security(_FIND)
        popen.assert_not_called()
        assert (tmp_path / "circuit.json").read_bytes() == original
        assert _circuit(tmp_path)["ts"] == ts
        assert _circuit(tmp_path)["reason"] == "dialog_busy"
        assert _log_rows(tmp_path)[-1]["reason"] == "circuit_open"

    clock["now"] = opened + ttl
    proc = _DoneProc(0)
    with patch("claude_swap.macos_keychain.subprocess.Popen", return_value=proc) as popen:
        result = macos_keychain._run_security(_FIND)
    popen.assert_called_once()
    assert result.returncode == 0
    assert _circuit(tmp_path)["open"] is False


def test_reset_with_no_circuit_file_succeeds(tmp_path, capsys):
    import sys

    from claude_swap import cli

    path = tmp_path / "circuit.json"
    assert not path.exists()
    assert macos_keychain.reset_keychain_circuit() is True
    assert _circuit(tmp_path)["open"] is False
    assert _stray_temps(tmp_path) == []

    path.unlink()
    assert not path.exists()
    with patch.object(sys, "argv", ["cswap", "keychain-circuit", "reset"]):
        cli.main()
    assert "Keychain spawn circuit reset (closed)" in capsys.readouterr().out
    assert _circuit(tmp_path)["open"] is False
    assert _stray_temps(tmp_path) == []


def test_keychain_circuit_reset_write_failure_exits_nonzero(tmp_path, monkeypatch, capsys):
    import sys

    from claude_swap import cli

    _write_circuit(tmp_path, open_=True, reason="dialog_busy")
    before = (tmp_path / "circuit.json").read_bytes()

    def _fail_replace(src, dst):
        assert Path(src).parent == Path(dst).parent
        assert Path(src).exists()
        raise OSError("simulated replace failure")

    monkeypatch.setattr(macos_keychain.os, "replace", _fail_replace)
    assert macos_keychain.reset_keychain_circuit() is False
    assert (tmp_path / "circuit.json").read_bytes() == before
    assert _stray_temps(tmp_path) == []

    with patch.object(sys, "argv", ["cswap", "keychain-circuit", "reset"]):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "failed to reset the Keychain spawn circuit" in captured.err
    assert "Keychain spawn circuit reset (closed)" not in captured.out
    assert (tmp_path / "circuit.json").read_bytes() == before
    assert _stray_temps(tmp_path) == []


def test_circuit_write_is_atomic_and_leaves_no_temp_files(tmp_path, monkeypatch):
    real_replace = macos_keychain.os.replace
    replaced: dict[str, str] = {}

    def _spy(src, dst):
        replaced["src"] = str(src)
        replaced["dst"] = str(dst)
        assert Path(src).parent == Path(dst).parent
        real_replace(src, dst)

    monkeypatch.setattr(macos_keychain.os, "replace", _spy)
    assert macos_keychain.reset_keychain_circuit() is True
    assert replaced["dst"].endswith("circuit.json")
    assert not Path(replaced["src"]).exists()
    assert _circuit(tmp_path)["open"] is False
    assert _stray_temps(tmp_path) == []

    # A failed replace must not publish over the previous file or leave a temp.
    before = (tmp_path / "circuit.json").read_bytes()

    def _fail_replace(src, dst):
        assert Path(src).exists()
        assert Path(src).parent == Path(dst).parent
        raise OSError("simulated replace failure")

    monkeypatch.setattr(macos_keychain.os, "replace", _fail_replace)
    assert macos_keychain._write_circuit(open_=True, reason="timeout") is False
    assert (tmp_path / "circuit.json").read_bytes() == before
    assert _stray_temps(tmp_path) == []


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
    monkeypatch.setattr(
        macos_keychain, "_dialog_busy", lambda exclude_pid=None: True, raising=False
    )
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
    monkeypatch.setattr(
        macos_keychain, "_dialog_busy", lambda exclude_pid=None: True, raising=False
    )
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
