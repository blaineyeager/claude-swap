"""macOS Keychain access via the ``security`` CLI.

A small wrapper around the system ``security`` tool for storing generic
passwords, used instead of the third-party ``keyring`` library. Two reasons:

- The macOS hot path no longer needs the ``keyring`` dependency.
- Keychain items are created and read by the same stable ``security`` binary, so
  reads stay silent across upgrades. ``keyring`` (and any in-process
  Security.framework call) anchors the item's access to the *Python interpreter*,
  which ``uv tool upgrade`` rebuilds — at which point macOS can show the "wants to
  use your keychain" prompt. ``security`` never changes, so creator == reader and
  there is no prompt.

The read/write/delete shapes mirror Claude Code's own implementation
(``utils/secureStorage/macOsKeychainStorage.ts``):

- ``set_password`` hex-encodes the value (``-X``) and pipes the command through
  ``security -i`` (stdin) so the secret never appears in process argv (a
  process-monitor / CrowdStrike concern). It falls back to argv only when the
  command would overflow ``security -i``'s 4096-byte stdin line buffer, which
  would otherwise truncate mid-argument and silently corrupt the entry.
- ``get_password`` uses ``find-generic-password ... -w`` and treats exit code 44
  as "not found" (returns ``None``); any *other* non-zero exit raises so callers
  can tell a genuine miss apart from a locked/denied/unavailable Keychain.

Caveat: values must be printable text. ``find-generic-password -w`` prints the
stored data raw only when it is printable; data with non-printable bytes comes
back *hex-encoded*, so a write/read round-trip would not be identity. Fine for
this codebase (credentials are ASCII JSON), but don't reuse this wrapper for
arbitrary binary data. Claude Code's ``-w`` reads share the same constraint.

This module is import-safe on every platform (it only shells out at call time);
its functions are only meaningful on macOS.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ``security -i`` reads stdin with a 4096-byte fgets() buffer (BUFSIZ on darwin).
# A command line longer than this is truncated mid-argument: it fails to write
# while leaving any previous entry intact (Claude Code #30337). 64 bytes of
# headroom guards against line-terminator accounting differences.
SECURITY_STDIN_LINE_LIMIT = 4096 - 64

_NOT_FOUND_RC = 44  # errSecItemNotFound surfaced by find/delete-generic-password

# Bound every ``security`` spawn so a wedged Keychain (a locked login keychain
# prompting for an unlock that never comes on a headless/SSH host) can't hang the
# CLI. 5s, deliberately short: a credential op that has to fall back to the file
# may be followed by a best-effort cleanup spawn, so the per-op budget doubles in
# the worst case. A healthy Keychain answers in well under 100ms.
_TIMEOUT = 5.0

# Pin the absolute path to Apple's system binary rather than resolving via PATH:
# this is a credential tool, so an attacker-controlled ``security`` earlier on
# PATH must not be able to intercept secrets. ``/usr/bin/security`` is present on
# every macOS.
_SECURITY = "/usr/bin/security"

# Overrides ``_TIMEOUT``. A host that can actually draw a Keychain dialog wants
# longer than 5s so a human gets a chance to answer before anything is killed;
# a headless one wants the short default. Garbage values fall back rather than
# meaning "no deadline" (a hang) or "kill instantly".
_TIMEOUT_ENV = "CLAUDE_SWAP_KEYCHAIN_TIMEOUT"

# How long a timed-out ``security`` gets to unwind after SIGTERM. If it is
# still alive after this, we leave it — SIGKILL is what aborts securityd.
_TERM_GRACE = 2.0

# JSONL breadcrumb for the next keychain-dialog incident. Override in tests.
# Never write secrets here: see ``_sanitize_security_argv``.
_LOG_ENV = "CLAUDE_SWAP_KEYCHAIN_LOG"

# Sticky spawn circuit. Open on dialog/timeout/single-flight refuse; close
# only on spawn rc 0 or 44, or :func:`reset_keychain_circuit`. Not a timer.
# ``no_keychain_env`` does not open it — that opt-out is process-local.
_CIRCUIT_ENV = "CLAUDE_SWAP_KEYCHAIN_CIRCUIT"

# A leftover ``security`` child from a prior timeout in this process. Never
# SIGKILL it; the next spawn refuses with ``single_flight`` while it lives.
_left_alive: object | None = None


def _keychain_log_path() -> Path:
    raw = os.environ.get(_LOG_ENV)
    if raw:
        return Path(raw)
    return Path.home() / ".claude" / "state" / "keychain-watch" / "cswap-keychain.jsonl"


def _sanitize_security_argv(argv: list[str]) -> list[str]:
    """Drop hex payloads (``-X``) so a timeout log cannot leak a credential."""
    out: list[str] = []
    redact_next = False
    for arg in argv:
        if redact_next:
            out.append("<redacted>")
            redact_next = False
            continue
        if arg == "-X":
            out.append(arg)
            redact_next = True
            continue
        if arg.startswith("-X") and arg != "-X":
            out.append("-X<redacted>")
            continue
        if arg == "-p":
            out.append(arg)
            redact_next = True
            continue
        out.append(arg)
    return out


def _caller_argv() -> list[str]:
    """cswap's own argv, truncated and stripped of anything that looks like a token."""
    out: list[str] = []
    for arg in sys.argv[:6]:
        if arg.startswith("sk-ant") or len(arg) > 80:
            out.append("<redacted>")
        else:
            out.append(arg)
    return out


def _log_security_event(payload: dict) -> None:
    """Append one JSON line. Must never raise — logging is best-effort."""
    try:
        path = _keychain_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **payload,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    except Exception:
        return


def _timeout() -> float:
    """The per-spawn deadline: :data:`_TIMEOUT`, or a sane ``_TIMEOUT_ENV``."""
    raw = os.environ.get(_TIMEOUT_ENV)
    if not raw:
        return _TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        return _TIMEOUT
    if not math.isfinite(value) or value <= 0:
        return _TIMEOUT
    return value


def _close_pipes(proc: subprocess.Popen[str]) -> None:
    """Drop our ends of a leftover child's pipes so *this* process does not leak FDs."""
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        if stream is None:
            continue
        try:
            stream.close()
        except OSError:
            pass


class KeychainError(Exception):
    """A ``security`` invocation failed for a reason other than "not found"."""


def _circuit_path() -> Path:
    raw = os.environ.get(_CIRCUIT_ENV)
    if raw:
        return Path(raw)
    return Path.home() / ".claude" / "state" / "keychain-watch" / "circuit.json"


def _write_circuit(*, open_: bool, reason: str) -> None:
    try:
        path = _circuit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "open": open_,
            "reason": reason,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        return


def _circuit_is_open() -> bool:
    try:
        data = json.loads(_circuit_path().read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("open") is True


def _open_circuit(reason: str) -> None:
    _write_circuit(open_=True, reason=reason)


def _close_circuit() -> None:
    _write_circuit(open_=False, reason="ok")


def reset_keychain_circuit() -> None:
    """Close the sticky spawn circuit. Does not signal any leftover child."""
    _close_circuit()


def _ps_text() -> str:
    """Process table snapshot. Never asks ``/usr/bin/security``."""
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,etime=,command="],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout or ""


def _dialog_busy(exclude_pid: int | None = None) -> bool:
    """True when SecurityAgent or a parked ``/usr/bin/security`` is visible.

    ``exclude_pid`` skips that process-table row so a timeout does not treat
    the child currently being waited on as an already-up dialog.
    """
    text = _ps_text()
    skip = None if exclude_pid is None else str(exclude_pid)
    for line in text.splitlines():
        if skip is not None:
            fields = line.split()
            if fields and fields[0] == skip:
                continue
        if "SecurityAgent" in line or "/usr/bin/security" in line:
            return True
    return False


def _single_flight_blocked() -> bool:
    global _left_alive
    proc = _left_alive
    if proc is None:
        return False
    poll = getattr(proc, "poll", None)
    try:
        rc = poll() if callable(poll) else getattr(proc, "returncode", None)
    except Exception:
        rc = None
    if rc is not None:
        _left_alive = None
        return False
    return True


def _record_left_alive(proc: object) -> None:
    global _left_alive
    _left_alive = proc


def _spawn_refuse_reason() -> str | None:
    if os.environ.get("CLAUDE_SWAP_NO_KEYCHAIN") == "1":
        return "no_keychain_env"
    if _circuit_is_open():
        return "circuit_open"
    if _dialog_busy():
        return "dialog_busy"
    if _single_flight_blocked():
        return "single_flight"
    return None


def _refuse_spawn(argv: list[str], reason: str) -> None:
    # Opt-out is process-env, not a durable host fault. Opening the circuit
    # here would keep refusing after the env is gone; reset has no prod caller.
    if reason != "no_keychain_env":
        _open_circuit(reason)
    _log_security_event(
        {
            "event": "security_spawn_refused",
            "reason": reason,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "argv": _sanitize_security_argv([str(a) for a in argv]),
            "cswap_argv": _caller_argv(),
            "no_keychain": os.environ.get("CLAUDE_SWAP_NO_KEYCHAIN") == "1",
        }
    )
    raise KeychainError(f"security spawn refused ({reason})")


def _run_security(
    argv: list[str],
    *,
    input: str | None = None,  # noqa: A002 - mirrors subprocess.run's name
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Spawn ``security``, sending SIGTERM on overrun — never SIGKILL.

    Drop-in for ``subprocess.run(..., capture_output=True, text=True)``: same
    :class:`subprocess.CompletedProcess` out, same :class:`subprocess.TimeoutExpired`
    on overrun, so every caller's error handling is unchanged.

    Refuses (raises :class:`KeychainError`, no Popen) when
    ``CLAUDE_SWAP_NO_KEYCHAIN=1``, the sticky circuit is open, SecurityAgent or
    a parked ``/usr/bin/security`` is in the process table, or a leftover child
    from a prior timeout is still alive. That gate lives here so session.py
    cannot bypass it.

    The one difference on a spawned timeout is the signal, and it is the whole
    point. ``subprocess.run(timeout=...)`` sends **SIGKILL**. When the child is
    parked on a SecurityAgent consent dialog that cannot be drawn (shielded
    screen, headless launchd job), SIGKILL makes it vanish with an XPC query
    still registered; ``securityd`` then destroys a still-held mutex tearing
    that query down, gets ``EBUSY``, and the uncaught ``Security::UnixError``
    aborts the daemon. Since the login keychain's master key lives only in that
    process's memory, the respawn comes up locked and every app on the machine
    re-prompts for the keychain password.

    If a dialog is already up at timeout, send **no signal** — SIGTERM is not
    processed until the dialog returns, and the old grace then SIGKILLed.
    Otherwise SIGTERM, then leave the child. Escalating to SIGKILL is the crash
    (measured 2026-08-26 and again 2026-09-02).
    """
    if timeout is None:
        timeout = _timeout()
    reason = _spawn_refuse_reason()
    if reason:
        _refuse_spawn(argv, reason)
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE if input is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
        dialog = False
        try:
            dialog = _dialog_busy(exclude_pid=proc.pid)
        except Exception:
            dialog = False
        left_alive = False
        if dialog:
            # Do not SIGTERM/SIGKILL a child parked on SecurityAgent.
            _close_pipes(proc)
            left_alive = True
            _record_left_alive(proc)
            _open_circuit("dialog_busy")
        else:
            proc.terminate()
            try:
                proc.communicate(timeout=_TERM_GRACE)
            except subprocess.TimeoutExpired:
                # Do not SIGKILL. A child this stuck is almost certainly blocked
                # in SecurityAgent; killing it is what aborts securityd.
                _close_pipes(proc)
                poll = getattr(proc, "poll", None)
                left_alive = (
                    poll() is None if callable(poll) else proc.returncode is None
                )
                if left_alive:
                    _record_left_alive(proc)
            _open_circuit("timeout")
        _log_security_event(
            {
                "event": "security_timeout",
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "child_pid": getattr(proc, "pid", None),
                "child_alive_after_term": left_alive,
                "timeout_s": timeout,
                "term_grace_s": 0 if dialog else _TERM_GRACE,
                "argv": _sanitize_security_argv([str(a) for a in argv]),
                "cswap_argv": _caller_argv(),
                "no_keychain": os.environ.get("CLAUDE_SWAP_NO_KEYCHAIN") == "1",
                "dialog_busy": dialog,
            }
        )
        raise
    if proc.returncode in (0, _NOT_FOUND_RC):
        _close_circuit()
    return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)


# Fallback if ``claude`` is not on PATH at write time (e.g. a launchd/cron job
# with a stripped PATH) — the standalone-CLI install location, matching most
# hosts' actual layout. ``_trusted_app_paths`` tries ``PATH`` first.
_CLAUDE_FALLBACK_PATH = os.path.expanduser("~/.local/bin/claude")


def _trusted_app_paths() -> list[str]:
    """Applications to grant no-prompt access to every Keychain item this module
    writes, via ``security add-generic-password -T``.

    Why this exists: ``set_password`` uses ``-U`` (update-or-create). Per
    ``security``'s own semantics, omitting ``-T`` on a write does NOT mean "leave
    the existing ACL alone" — it means "trust only the process performing this
    write" (``/usr/bin/security`` itself here). That's fine for *this* module's
    own reads (see the module docstring: creator == reader == ``security``), but
    Claude Code reads the same "Claude Code-credentials" item **in-process** via
    its own Security.framework call, a different, untrusted app. Every ``cswap``
    rotation was silently re-pinning the ACL to "security only," so Claude Code's
    next read had to fall back to a GUI "wants to use your keychain" prompt — one
    that never gets answered in a headless/launchd daemon, surfacing as "OAuth
    session expired and could not be refreshed" even though the credential itself
    (and its ``expiresAt``) was fine.

    Deliberately NEVER pass ``-T ""`` — that is macOS's own footgun: an *empty*
    ``-T`` means "trust every application," not "trust none."

    Trust here is anchored to the binary's **code signature** (Developer ID cert
    + Team ID + bundle identifier), not the literal file path — that's how macOS
    keychain ACLs evaluate a trusted-application entry. So resolving ``claude``
    via ``PATH``/``shutil.which`` each call (rather than hardcoding today's
    version-pinned binary under ``~/.local/share/claude/versions/<X.Y.Z>``, which
    a self-update deletes) is both simpler AND durable: the grant is computed
    from whatever the live, currently-signed ``claude`` binary is at write time,
    and continues to match future Anthropic-signed builds of the same identity.

    ``/usr/bin/security`` is included explicitly: specifying ``-T`` at all drops
    the implicit "trust the calling process" default, so without listing it here
    this module's OWN future reads (via ``get_password``) would themselves start
    prompting.

    Does NOT include the ``cswap`` Python interpreter — this module deliberately
    never touches Keychain items except through the ``security`` CLI (that's the
    whole point of the module docstring's "creator == reader" design), so a
    separate grant for the venv's ``python3`` would be redundant and, being a
    ``uv``-managed venv path, less stable than ``/usr/bin/security`` itself.
    """
    paths = [_SECURITY]
    claude_path = shutil.which("claude") or (
        _CLAUDE_FALLBACK_PATH if os.path.exists(_CLAUDE_FALLBACK_PATH) else None
    )
    if claude_path:
        paths.append(claude_path)
    return paths


# The exceptions a Keychain operation may raise that callers should treat as
# "Keychain unusable" (→ fall back to file storage) rather than a programming
# bug: a wrapper failure (KeychainError, incl. a converted timeout), a raw
# subprocess timeout, or a missing ``security`` binary (OSError). Catching this
# tuple — never bare ``Exception`` — keeps a real bug loud instead of silently
# routing to the file backend mid-invocation.
KEYCHAIN_ERRORS = (KeychainError, subprocess.TimeoutExpired, OSError)


def keychain_account_name() -> str:
    """Account name for the active-credential Keychain item, mirroring Claude
    Code's ``getUsername()`` (``utils/secureStorage/macOsKeychainHelpers.ts``).

    ``$USER`` first, then the OS username, then a stable final fallback. Matching
    this exactly matters on headless/launchd/cron hosts where ``$USER`` is unset:
    a divergent default (e.g. ``"user"``) would key a *different* Keychain item
    than Claude Code, so the two could not see each other's active credential.
    """
    user = os.environ.get("USER")
    if user:
        return user
    try:
        import pwd  # POSIX-only; the account-name call sites are macOS-only

        return pwd.getpwuid(os.geteuid()).pw_name
    except Exception:
        return "claude-code-user"


def _quote(value: str) -> str:
    """Quote a value for a ``security -i`` stdin command line.

    ``security -i`` re-parses each line shell-style, so wrap the value in double
    quotes and backslash-escape any embedded ``"``/``\\`` (e.g. the active-
    credential service name contains a space).
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def get_password(service: str, account: str) -> str | None:
    """Return the stored password, or ``None`` if no such item exists (rc 44).

    Raises :class:`KeychainError` on any other non-zero exit (locked / denied /
    unavailable) or a timeout, so a genuine miss is not confused with a transient
    failure.
    """
    try:
        result = _run_security(
            [_SECURITY, "find-generic-password", "-a", account, "-w", "-s", service],
            timeout=_timeout(),
        )
    except subprocess.TimeoutExpired as e:
        raise KeychainError(
            f"security find-generic-password timed out after {_TIMEOUT}s"
        ) from e
    if result.returncode == 0:
        # `-w` prints the value followed by one newline; strip exactly that so
        # values with meaningful leading/trailing whitespace survive intact.
        return result.stdout.removesuffix("\n")
    if result.returncode == _NOT_FOUND_RC:
        return None
    raise KeychainError(
        f"security find-generic-password failed (rc={result.returncode}): "
        f"{result.stderr.strip()}"
    )


def item_exists(service: str, account: str) -> bool:
    """Whether a generic-password item exists, without touching its secret.

    Attribute-only lookup (no ``-w``): nothing is decrypted, so this can never
    trigger a Keychain prompt, even for items owned by another app. Returns
    ``True`` only on rc 0; "not found" (rc 44), error exits, a timeout, a
    refused spawn, and a missing binary all return ``False``. Deliberately
    **non-raising**: callers use it for cleanup verification, not access
    decisions, so it must never feed the capability cache (a timeout here
    means "couldn't tell", not "Keychain works").
    """
    try:
        result = _run_security(
            [_SECURITY, "find-generic-password", "-a", account, "-s", service],
            timeout=_timeout(),
        )
    except (subprocess.TimeoutExpired, OSError, KeychainError):
        return False
    return result.returncode == 0


def set_password(service: str, account: str, password: str) -> None:
    """Create or update a generic-password item (``-U``).

    Prefers ``security -i`` stdin so the secret stays out of argv; falls back to
    argv only for payloads that would overflow the stdin line buffer. Raises
    :class:`KeychainError` on a non-zero exit or a timeout.

    Every write stamps the trusted-application ACL via ``-T`` (see
    :func:`_trusted_app_paths`) so the item stays readable by Claude Code itself,
    not just by this module — ``-U`` otherwise resets the ACL to "creator only"
    on every call, which silently locked Claude Code out after each rotation.
    """
    hex_value = password.encode("utf-8").hex()
    trust_args = "".join(f"-T {_quote(p)} " for p in _trusted_app_paths())
    # `-X` passes the value as hex, avoiding any escaping issues for the secret.
    command = (
        f"add-generic-password -U -a {_quote(account)} -s {_quote(service)} "
        f"{trust_args}-X {hex_value}\n"
    )
    try:
        if len(command.encode("utf-8")) <= SECURITY_STDIN_LINE_LIMIT:
            result = _run_security(
                [_SECURITY, "-i"],
                input=command,
                timeout=_timeout(),
            )
        else:
            # Overflows the stdin line buffer; fall back to argv. Hex in argv is
            # recoverable by a determined observer but defeats naive plaintext-grep
            # rules, and the alternative — silent corruption — is strictly worse.
            argv = [_SECURITY, "add-generic-password", "-U", "-a", account, "-s", service]
            for trusted_path in _trusted_app_paths():
                argv += ["-T", trusted_path]
            argv += ["-X", hex_value]
            result = _run_security(
                argv,
                timeout=_timeout(),
            )
    except subprocess.TimeoutExpired as e:
        raise KeychainError(
            f"security add-generic-password timed out after {_TIMEOUT}s"
        ) from e
    if result.returncode != 0:
        raise KeychainError(
            f"security add-generic-password failed (rc={result.returncode}): "
            f"{result.stderr.strip()}"
        )


def delete_password(service: str, account: str) -> None:
    """Delete a generic-password item. rc 44 (already absent) counts as success.

    Raises :class:`KeychainError` on any other non-zero exit or a timeout.
    """
    try:
        result = _run_security(
            [_SECURITY, "delete-generic-password", "-a", account, "-s", service],
            timeout=_timeout(),
        )
    except subprocess.TimeoutExpired as e:
        raise KeychainError(
            f"security delete-generic-password timed out after {_TIMEOUT}s"
        ) from e
    if result.returncode in (0, _NOT_FOUND_RC):
        return
    raise KeychainError(
        f"security delete-generic-password failed (rc={result.returncode}): "
        f"{result.stderr.strip()}"
    )
