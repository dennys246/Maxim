"""Console formatting utilities with ANSI colors.

Provides consistent, color-coded terminal output for diagnostics,
status messages, and other CLI output across the codebase.
"""

from __future__ import annotations

import sys


class Colors:
    """ANSI color codes for terminal output.

    Usage:
        print(f"{Colors.GREEN}Success!{Colors.END}")
        print(f"{Colors.BOLD}{Colors.RED}Error{Colors.END}")
    """

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"

    @classmethod
    def is_supported(cls) -> bool:
        """Check if terminal supports ANSI colors."""
        # Check if stdout is a TTY
        if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
            return False
        # Check for known color-supporting terminals
        import os
        term = os.environ.get("TERM", "")
        return term != "dumb" and "color" in term or term in ("xterm", "screen", "vt100")


# ─────────────────────────────────────────────────────────────────────────────
# Formatted Print Functions
# ─────────────────────────────────────────────────────────────────────────────


def print_header(text: str, char: str = "=", width: int = 70) -> None:
    """Print a formatted section header.

    Args:
        text: Header text to display.
        char: Character to use for the divider line.
        width: Width of the divider line.
    """
    divider = char * width
    print(f"\n{Colors.BOLD}{Colors.BLUE}{divider}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{divider}{Colors.END}\n")


def print_success(text: str, prefix: str = "✓") -> None:
    """Print a success message in green.

    Args:
        text: Message to display.
        prefix: Prefix character/emoji (default: checkmark).
    """
    print(f"{Colors.GREEN}{prefix} {text}{Colors.END}")


def print_warning(text: str, prefix: str = "⚠") -> None:
    """Print a warning message in yellow.

    Args:
        text: Message to display.
        prefix: Prefix character/emoji (default: warning sign).
    """
    print(f"{Colors.YELLOW}{prefix} {text}{Colors.END}")


def print_error(text: str, prefix: str = "✗") -> None:
    """Print an error message in red.

    Args:
        text: Message to display.
        prefix: Prefix character/emoji (default: X mark).
    """
    print(f"{Colors.RED}{prefix} {text}{Colors.END}")


def print_info(text: str, indent: int = 2) -> None:
    """Print an info message with optional indentation.

    Args:
        text: Message to display.
        indent: Number of spaces to indent.
    """
    print(f"{' ' * indent}{text}")


def print_status(label: str, value: str, ok: bool = True) -> None:
    """Print a status line with colored value.

    Args:
        label: Status label.
        value: Status value.
        ok: Whether status is OK (green) or not (red).
    """
    color = Colors.GREEN if ok else Colors.RED
    print(f"  {label}: {color}{value}{Colors.END}")


def print_dim(text: str) -> None:
    """Print dimmed/muted text.

    Args:
        text: Message to display.
    """
    print(f"{Colors.DIM}{text}{Colors.END}")


def colorize(text: str, color: str) -> str:
    """Wrap text with ANSI color codes.

    Args:
        text: Text to colorize.
        color: Color attribute from Colors class (e.g., Colors.GREEN).

    Returns:
        Colorized string.
    """
    return f"{color}{text}{Colors.END}"


__all__ = [
    "Colors",
    "print_header",
    "print_success",
    "print_warning",
    "print_error",
    "print_info",
    "print_status",
    "print_dim",
    "colorize",
]
