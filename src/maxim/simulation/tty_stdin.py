"""The simulation's termios/TTY raw-mode read loops, moved verbatim out of ``orchestrator.py``.

WHAT MAY LIVE HERE: ONLY the termios/TTY read loops of
``orchestrator.py::start_simulation_mode`` and the module state that exists only for them
(the escape-sequence keymap ``_KEYMAP`` and the terminal-restore record ``_termios_restore``
with its atexit/SIGTERM restore hook, moved with the loop per decomposition rule (a)). Any other
code belongs in ``orchestrator.py`` or another measured module. This file is on the coverage
exclusion list (``scripts/coverage_exclusions.json``), with a ``excluded_statements`` ledger entry
pinning its AST statement count, because every line below needs a real TTY on stdin. The test
harness never runs one (owner decisions (e) and SF-4, 2026-10-13); see that entry's reason.

The two functions were nested in ``start_simulation_mode`` (1.3.2 orchestrator decomposition,
slice 0) and moved byte-for-byte apart from the closure-to-parameter rewrite (rule (d)): each
closure variable the body reads is now a keyword-only parameter of the SAME name, so the body
text did not change. The one closure rebind, ``_new_goal`` in the post-sim reader, became the
return value. One later edit (2026-10-26, #1454 interim): the post-sim prompt's text dropped its false
"memory carries over" promise, as the orchestrator's panel prompt did.

  - ``stdin_reader_raw``: the ``sim.stdin`` thread's target in interactive mode. It still falls
    back to the caller's ``_stdin_reader_line`` when stdin is not a TTY, termios is missing or
    ``tcgetattr`` fails.
  - ``read_post_sim_goal_raw``: the post-sim "new goal or Enter" prompt, called only when stdin
    is a TTY.
"""

from __future__ import annotations

import atexit
import signal
import threading
from typing import Any, Callable

# Module-level storage for terminal restore on ungraceful exit.
# Set by stdin_reader_raw; read by the atexit/SIGTERM handler.
_termios_restore: tuple[int, list] | None = None

# Declarative keymap: escape sequence suffix -> MaximDisplay method name.
# Plain arrows (3-byte: \x1b + "[X"), shift+arrows (6-byte: \x1b + "[1;2X"),
# option/alt+arrows (6-byte: \x1b + "[1;3X").
_KEYMAP: dict[str, str] = {
    # Plain arrows — log scroll
    "[A": "scroll_up",
    "[B": "scroll_down",
    "[C": "scroll_bottom",
    "[D": "scroll_page_up",
    # Shift+arrows — thinking scroll + agent focus
    "[1;2A": "scroll_thinking_up",
    "[1;2B": "scroll_thinking_down",
    "[1;2C": "focus_next",
    "[1;2D": "focus_prev",
    # Option (Alt)+arrows — layout resize
    "[1;3A": "resize_thinking_more",
    "[1;3B": "resize_thinking_less",
}


def _restore_terminal() -> None:
    """Restore terminal settings if raw mode was enabled. atexit + SIGTERM safe."""
    global _termios_restore
    if _termios_restore is not None:
        fd, old_settings = _termios_restore
        _termios_restore = None
        try:
            import termios

            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass


def _sigterm_handler(signum: int, frame: Any) -> None:
    """Restore terminal on SIGTERM, then exit."""
    _restore_terminal()
    raise SystemExit(128 + signum)


atexit.register(_restore_terminal)


def stdin_reader_raw(
    *,
    stop_event: threading.Event,
    operator_stop: threading.Event,
    _sim_prompt_handler: Any,
    _process_line: Callable[[str], bool],
    _stdin_reader_line: Callable[[], None],
    _stop_by_operator: Callable[[threading.Event, threading.Event], None],
    get_active_display: Callable[[], Any],
) -> None:
    """Character-level stdin reader using raw terminal mode.

    Disables echo so keystrokes don't appear at the raw cursor
    (below the Rich Live panel). Instead, typed characters are
    rendered inside the display's input panel via set_prompt().
    On Enter, the buffered line is processed normally.
    """
    import sys

    stdin = sys.stdin
    if stdin is None or not hasattr(stdin, "isatty") or not stdin.isatty():
        # Non-TTY — fall back to line-based reader
        _stdin_reader_line()
        return

    try:
        import select
        import termios
        import tty
    except ImportError:
        _stdin_reader_line()
        return

    try:
        fd = stdin.fileno()
        old_settings = termios.tcgetattr(fd)
    except Exception:
        _stdin_reader_line()
        return

    # Store for atexit/SIGTERM restoration in case of ungraceful exit.
    global _termios_restore
    _termios_restore = (fd, old_settings)
    # Install SIGTERM handler (SIGINT is handled by Ctrl+C in raw mode).
    try:
        signal.signal(signal.SIGTERM, _sigterm_handler)
    except (OSError, ValueError):
        pass  # Not main thread or unsupported platform

    buf: list[str] = []

    def _update_prompt() -> None:
        display = get_active_display()
        if display is not None:
            text = "".join(buf)
            if _sim_prompt_handler is not None and _sim_prompt_handler.has_pending_prompt:
                # Show the agent's question + user's typing
                prompt_text = _sim_prompt_handler.pending_display_text
                display.set_prompt(f"{prompt_text}\n\n> {text}")
            else:
                display.set_prompt(f"> {text}")

    try:
        tty.setcbreak(fd)
        new_settings = termios.tcgetattr(fd)
        new_settings[3] &= ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)

        _update_prompt()

        import os as _os_mod

        def _read_char() -> str:
            """Read a single UTF-8 character from stdin using raw OS read.

            Detects multi-byte UTF-8 leading bytes and reads the
            correct number of continuation bytes so characters like
            ≠ (3 bytes) arrive as one Python str, not three garbled
            replacement chars.
            """
            b = _os_mod.read(fd, 1)
            if not b:
                return ""
            first = b[0]
            if first < 0x80:
                return b.decode("utf-8", errors="replace")
            # Multi-byte UTF-8: leading byte tells us how many follow
            if 0xC0 <= first < 0xE0:
                extra = 1
            elif 0xE0 <= first < 0xF0:
                extra = 2
            elif 0xF0 <= first < 0xF8:
                extra = 3
            else:
                return b.decode("utf-8", errors="replace")
            rest = _os_mod.read(fd, extra)
            return (b + rest).decode("utf-8", errors="replace")

        while not stop_event.is_set():
            try:
                ready, _, _ = select.select([stdin], [], [], 0.1)
            except Exception:
                continue
            if not ready:
                continue
            try:
                ch = _read_char()
            except Exception:
                continue
            if not ch:
                continue

            if ch in ("\n", "\r"):
                # Process the buffered line
                line = "".join(buf)
                buf.clear()
                _update_prompt()
                if _process_line(line):
                    break
            elif ch == "\x7f" or ch == "\x08":
                # Backspace
                if buf:
                    buf.pop()
                _update_prompt()
            elif ch == "\x15":
                # Ctrl+U — clear line
                buf.clear()
                _update_prompt()
            elif ch == "\x1b":
                # Escape sequence — variable-length accumulator.
                # Plain arrows: \x1b[A (3 bytes). Shift+arrows:
                # \x1b[1;2A (6 bytes). Ctrl+arrows: \x1b[1;5A (6 bytes).
                # Read byte-by-byte until a terminator (A-Z, a-z, ~).
                try:
                    # Guard with select to avoid blocking on lone Escape.
                    _esc_ready, _, _ = select.select([stdin], [], [], 0.05)
                    if not _esc_ready:
                        continue  # Lone Escape — discard
                    _seq_bytes = bytearray()
                    for _ in range(8):  # max escape sequence length
                        _sr, _, _ = select.select([stdin], [], [], 0.01)
                        if not _sr:
                            break
                        _b = _os_mod.read(fd, 1)
                        if not _b:
                            break
                        _seq_bytes.extend(_b)
                        _c = _b[0]
                        if (0x41 <= _c <= 0x5A) or (0x61 <= _c <= 0x7A) or _c == 0x7E:
                            break
                    seq = _seq_bytes.decode("ascii", errors="replace")
                    _action = _KEYMAP.get(seq)
                    if _action:
                        display = get_active_display()
                        if display is not None:
                            _method = getattr(display, _action, None)
                            if _method:
                                _method()
                except Exception:
                    pass
            elif ch == "\x03":
                # Ctrl+C
                _stop_by_operator(stop_event, operator_stop)
                break
            elif ch == "\x04":
                # Ctrl+D (EOF)
                return
            elif ch == "\u2260":
                # ≠ (Option+= on macOS) — resize: more thinking
                display = get_active_display()
                if display is not None:
                    display.resize_thinking_more()
            elif ch == "\u2013":
                # – (Option+- on macOS) — resize: less thinking
                display = get_active_display()
                if display is not None:
                    display.resize_thinking_less()
            elif ch >= " " or ch == "\t":
                # Printable character
                buf.append(ch)
                _update_prompt()
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass
        _termios_restore = None  # Restored — no atexit needed
        display = get_active_display()
        if display is not None:
            display.clear_prompt()


def read_post_sim_goal_raw(*, _stdin: Any, display: Any) -> str:
    """The post-sim prompt's raw read loop: the typed goal, or "" on Enter alone, Ctrl+C or a failure."""
    _new_goal = ""
    try:
        import os as _os3
        import select as _sel3
        import termios as _term3
        import tty as _tty3

        _fd3 = _stdin.fileno()
        _old3 = _term3.tcgetattr(_fd3)
        _buf3: list[str] = []
        try:
            _tty3.setcbreak(_fd3)
            _new3 = _term3.tcgetattr(_fd3)
            _new3[3] &= ~_term3.ECHO
            _term3.tcsetattr(_fd3, _term3.TCSADRAIN, _new3)

            while True:
                ready, _, _ = _sel3.select([_stdin], [], [], 0.1)
                if not ready:
                    continue
                ch = _os3.read(_fd3, 1).decode("utf-8", errors="replace")
                if ch in ("\n", "\r"):
                    _new_goal = "".join(_buf3).strip()
                    break
                elif ch == "\x7f" or ch == "\x08":
                    if _buf3:
                        _buf3.pop()
                elif ch == "\x15":
                    _buf3.clear()
                elif ch == "\x1b":
                    rest = _os3.read(_fd3, 2)
                    if len(rest) == 2 and display is not None:
                        seq = rest.decode("ascii", errors="replace")
                        if seq == "[A":
                            display.scroll(3)
                        elif seq == "[B":
                            display.scroll(-3)
                        elif seq == "[C":
                            display.scroll(-999999)
                        elif seq == "[D":
                            approx = display.page_height
                            display.scroll(approx)
                elif ch == "\x03":
                    break
                elif ch >= " ":
                    _buf3.append(ch)

                # Update prompt with typed text
                if display is not None:
                    typed = "".join(_buf3)
                    display.set_prompt(f"Tell me a new simulation to imagine, or press Enter to finish.\n\n> {typed}")
        finally:
            _term3.tcsetattr(_fd3, _term3.TCSADRAIN, _old3)
    except Exception:
        pass
    return _new_goal
