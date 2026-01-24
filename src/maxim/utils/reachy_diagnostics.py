#!/usr/bin/env python3
"""
Reachy Mini Connection Diagnostics

Tests connectivity to a Reachy Mini robot and diagnoses common connection issues.
Run this script to verify network connectivity and check if required services are running.

Usage:
    python -m maxim.utils.reachy_diagnostics [--host REACHY_IP]
    python -m maxim.utils.reachy_diagnostics --host 192.168.50.149
"""

import argparse
import os
import socket
import subprocess
import sys
from typing import Tuple


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str) -> None:
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}\n")


def print_success(text: str) -> None:
    """Print success message in green"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_warning(text: str) -> None:
    """Print warning message in yellow"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_error(text: str) -> None:
    """Print error message in red"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text: str) -> None:
    """Print info message"""
    print(f"  {text}")


def test_ping(host: str, timeout: int = 3) -> Tuple[bool, str]:
    """
    Test basic network connectivity using ping

    Args:
        host: IP address or hostname to ping
        timeout: Timeout in seconds

    Returns:
        Tuple of (success, message)
    """
    try:
        # Use ping command with timeout
        result = subprocess.run(
            ['ping', '-c', '3', '-W', str(timeout), host],
            capture_output=True,
            text=True,
            timeout=timeout + 2
        )

        if result.returncode == 0:
            # Parse average latency from output
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'rtt min/avg/max' in line or 'round-trip' in line:
                    # Extract average latency
                    parts = line.split('=')
                    if len(parts) > 1:
                        stats = parts[1].strip().split('/')
                        if len(stats) >= 2:
                            avg_latency = stats[1]
                            return True, f"Average latency: {avg_latency} ms"
            return True, "Network reachable"
        else:
            return False, "Host unreachable"

    except subprocess.TimeoutExpired:
        return False, f"Ping timeout after {timeout}s"
    except FileNotFoundError:
        return False, "ping command not found"
    except Exception as e:
        return False, f"Error: {str(e)}"


def test_port(host: str, port: int, timeout: int = 5) -> Tuple[bool, str]:
    """
    Test if a TCP port is open and accepting connections

    Args:
        host: IP address or hostname
        port: Port number to test
        timeout: Connection timeout in seconds

    Returns:
        Tuple of (success, message)
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)

    try:
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            return True, "Port is open and accepting connections"
        else:
            return False, "Connection refused"

    except socket.timeout:
        return False, f"Connection timeout after {timeout}s"
    except socket.gaierror:
        return False, "Hostname could not be resolved"
    except Exception as e:
        return False, f"Error: {str(e)}"


def diagnose_reachy(host: str) -> int:
    """
    Run complete diagnostic check on Reachy Mini

    Args:
        host: Reachy IP address or hostname

    Returns:
        Exit code (0 = all tests passed, 1 = some tests failed)
    """
    print_header(f"Reachy Mini Connection Diagnostics")
    print_info(f"Target: {host}")
    print_info(f"Time: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}")

    all_passed = True

    # Test 1: Basic network connectivity
    print_header("Test 1: Basic Network Connectivity")
    success, message = test_ping(host)
    if success:
        print_success(f"Ping to {host}: {message}")
    else:
        print_error(f"Ping to {host}: {message}")
        print_info("Solution: Check that Reachy is powered on and connected to the same network")
        all_passed = False
        # If ping fails, no point testing ports
        return 1

    # Test 2: Zenoh port (motor control)
    print_header("Test 2: Zenoh Port (Motor Control)")
    success, message = test_port(host, 7447)
    if success:
        print_success(f"Port 7447 (Zenoh): {message}")
        print_info("Motor control communication is available")
    else:
        print_error(f"Port 7447 (Zenoh): {message}")
        print_info("Solution: Ensure the Reachy daemon is running")
        print_info("  SSH into Reachy: ssh pollen@{}")
        print_info("  Check daemon: systemctl status reachy-mini-daemon")
        all_passed = False

    # Test 3: Dashboard/API port
    print_header("Test 3: Dashboard/API Port")
    success, message = test_port(host, 8000)
    if success:
        print_success(f"Port 8000 (Dashboard): {message}")
        print_info(f"Dashboard available at: http://{host}:8000")
    else:
        print_error(f"Port 8000 (Dashboard): {message}")
        print_info("Solution: Check if the Reachy web dashboard is running")
        all_passed = False

    # Test 4: WebRTC signaling port
    print_header("Test 4: WebRTC Signaling Port")
    success, message = test_port(host, 8443)
    if success:
        print_success(f"Port 8443 (WebRTC): {message}")
        print_info("WebRTC media streaming is available")
    else:
        print_warning(f"Port 8443 (WebRTC): {message}")
        print_info("WebRTC signaling server is not running")
        print_info("Solutions:")
        print_info("  1. Restart the Reachy (press OFF, wait 5s, press ON)")
        print_info("  2. Use media_backend='gstreamer' or 'default' in your code")
        print_info("  3. SSH into Reachy and run: reachyminios_check")
        print_info("Note: This is optional - motor control will work without WebRTC")

    # Summary
    print_header("Diagnostic Summary")
    if all_passed:
        print_success("All critical services are accessible")
        print_info("Your Reachy Mini should connect successfully")
        return 0
    else:
        print_warning("Some services are not accessible")
        print_info("Review the solutions above to fix connection issues")
        return 1


def main() -> int:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Diagnose Reachy Mini connection issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect Reachy IP from environment or use default
  python -m maxim.utils.reachy_diagnostics

  # Specify Reachy IP address
  python -m maxim.utils.reachy_diagnostics --host 192.168.50.149

  # Use hostname
  python -m maxim.utils.reachy_diagnostics --host reachy-mini.local
        """
    )

    parser.add_argument(
        '--host',
        type=str,
        default=None,
        help='Reachy IP address or hostname (default: $MAXIM_REACHY_HOST or 192.168.50.149)'
    )

    args = parser.parse_args()

    # Determine Reachy host
    host = args.host or os.getenv('MAXIM_REACHY_HOST', '192.168.50.149')

    try:
        return diagnose_reachy(host)
    except KeyboardInterrupt:
        print("\n\nDiagnostics interrupted by user")
        return 130
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
