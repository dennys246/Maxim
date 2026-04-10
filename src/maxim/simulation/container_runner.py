"""Container job runners — slurm/bsub-style wrapper abstraction.

Takes a job spec (image, memory, cpus, network, env, mounts, user,
command) and manages the container lifecycle. The ``ContainerRunner``
protocol is backend-agnostic so cloud providers (AWS Fargate/ECS,
Google Cloud Run, Azure Container Instances) can slot in by
implementing the protocol — the rest of Maxim depends only on the
abstract interface.

Shipped implementations:
- ``LocalDockerRunner``: uses the local ``docker`` CLI via subprocess.
  Zero new dependencies. Crash-safe via atexit cleanup hook.

Extension points (not yet implemented):
- ``FargateRunner``: AWS ECS/Fargate via boto3
- ``CloudRunRunner``: Google Cloud Run via google-cloud-run SDK
- ``AzureContainerInstancesRunner``: Azure ACI via azure-mgmt-containerinstance

Workspace storage is currently coupled to the runner (LocalDockerRunner
uses bind mounts to a host tmpdir). Cloud runners will need a separate
``WorkspaceProvider`` abstraction (EFS for Fargate, GCS/Filestore for
Cloud Run, Azure Files for ACI) — factored out when the first cloud
runner lands.

Design goals:
- Testable: docker invocations go through one ``_run_docker`` method
  that can be mocked.
- Crash-safe: an atexit hook is registered on first launch to guarantee
  cleanup even if the parent process is killed.
- Agent-friendly: the API takes a spec and returns a handle, making it
  trivial to later expose as an agent-facing tool
  (``launch_container(image=..., memory=..., ...)``).
"""

from __future__ import annotations

import atexit
import logging
import subprocess
import uuid
import weakref
from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContainerSpec:
    """Specification for a container job — the "submit a job" payload.

    This is intentionally similar to a batch-scheduler job spec. Pass it
    to ``ContainerRunner.launch()`` and get back a ``ContainerHandle``.
    """

    image: str = "python:3.12-slim"
    memory: str = "512m"  # Docker-style: "512m", "2g"
    cpus: float = 1.0  # CPU cores (can be fractional)
    network: str = "none"  # "none" | "bridge" | "host" | custom
    pids_limit: int = 64
    env: dict[str, str] = field(default_factory=dict)
    mounts: list[tuple[str, str, str]] = field(default_factory=list)
    # mounts: list of (host_path, container_path, mode) — mode is "rw" or "ro"
    user: str = "1000:1000"  # UID:GID inside container
    command: tuple[str, ...] = ("sleep", "infinity")
    workdir: str = "/home/user/.maxim_workspace"
    name_prefix: str = "maxim-sandbox"
    shell: str = "/bin/bash"  # "/bin/bash" or "/bin/sh" for Alpine/BusyBox


# ─────────────────────────────────────────────────────────────────────────────
# Image catalog — realistic deployment targets for Maxim
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ImageProfile:
    """Describes a Docker image Maxim could realistically be deployed on."""

    image: str
    family: str  # "debian", "ubuntu", "rhel", "alpine", "python"
    description: str
    has_useradd: bool  # whether ``useradd`` is available out of the box
    has_bash: bool  # whether /bin/bash exists (Alpine uses /bin/sh)
    size_mb_approx: int  # approximate compressed image size


IMAGE_CATALOG: dict[str, ImageProfile] = {
    # Python-focused (default — matches Maxim's own runtime)
    "python:3.12-slim": ImageProfile(
        "python:3.12-slim",
        "debian",
        "Debian-slim with Python 3.12. Default Maxim sandbox.",
        has_useradd=True,
        has_bash=True,
        size_mb_approx=45,
    ),
    "python:3.12-bookworm": ImageProfile(
        "python:3.12-bookworm",
        "debian",
        "Full Debian 12 with Python 3.12 + dev tools.",
        has_useradd=True,
        has_bash=True,
        size_mb_approx=350,
    ),
    # Ubuntu — most common Linux target for robots + laptops
    "ubuntu:22.04": ImageProfile(
        "ubuntu:22.04",
        "ubuntu",
        "Ubuntu 22.04 LTS (Jammy). Widely deployed on robotics platforms.",
        has_useradd=True,
        has_bash=True,
        size_mb_approx=30,
    ),
    "ubuntu:24.04": ImageProfile(
        "ubuntu:24.04",
        "ubuntu",
        "Ubuntu 24.04 LTS (Noble). Current Ubuntu LTS.",
        has_useradd=True,
        has_bash=True,
        size_mb_approx=30,
    ),
    # Debian — minimal, stable
    "debian:12-slim": ImageProfile(
        "debian:12-slim",
        "debian",
        "Debian 12 (Bookworm) slim. Minimal footprint for embedded use.",
        has_useradd=True,
        has_bash=True,
        size_mb_approx=30,
    ),
    # Red Hat family
    "rockylinux:9": ImageProfile(
        "rockylinux:9",
        "rhel",
        "Rocky Linux 9. Binary-compatible with RHEL 9.",
        has_useradd=True,
        has_bash=True,
        size_mb_approx=75,
    ),
    "almalinux:9": ImageProfile(
        "almalinux:9",
        "rhel",
        "AlmaLinux 9. Binary-compatible with RHEL 9.",
        has_useradd=True,
        has_bash=True,
        size_mb_approx=75,
    ),
    "registry.access.redhat.com/ubi9/ubi-minimal": ImageProfile(
        "registry.access.redhat.com/ubi9/ubi-minimal",
        "rhel",
        "Red Hat Universal Base Image 9 (minimal). Official RHEL base.",
        has_useradd=True,
        has_bash=True,
        size_mb_approx=35,
    ),
    # Alpine — minimal for edge / embedded
    "alpine:3.19": ImageProfile(
        "alpine:3.19",
        "alpine",
        "Alpine Linux 3.19. Minimal footprint (musl libc, no bash).",
        has_useradd=False,
        has_bash=False,
        size_mb_approx=8,
    ),
}


DEFAULT_IMAGE = "python:3.12-slim"


def get_image_profile(image: str) -> ImageProfile | None:
    """Look up metadata for a catalog image, if known."""
    return IMAGE_CATALOG.get(image)


def list_supported_images() -> list[str]:
    """Return the list of catalog image names."""
    return list(IMAGE_CATALOG.keys())


@dataclass(eq=False)
class ContainerHandle:
    """Handle to a running container — returned by ``launch()``.

    Identity-based hashing (``eq=False``) so handles can be tracked
    in weak sets for atexit cleanup.
    """

    container_id: str
    name: str
    spec: ContainerSpec


@dataclass
class ContainerExecResult:
    """Result of executing a command inside a container."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────


class ContainerRunnerError(RuntimeError):
    """Raised when a container runtime invocation fails unexpectedly."""


# ─────────────────────────────────────────────────────────────────────────────
# Protocol — the cloud-portable surface
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class ContainerRunner(Protocol):
    """Abstract interface for launching container jobs.

    Implementations include:
    - ``LocalDockerRunner`` (ships today)
    - ``FargateRunner`` / ``CloudRunRunner`` / ``AzureContainerInstancesRunner``
      (future — slot in by implementing this protocol)

    All methods must be safe to call from the main thread; thread-safety
    of concurrent calls is implementation-defined. Implementations
    should register their own atexit cleanup for crash safety.
    """

    def launch(self, spec: "ContainerSpec") -> "ContainerHandle":
        """Start a container from ``spec`` and return a handle."""
        ...

    def exec(
        self,
        handle: "ContainerHandle",
        command: str,
        *,
        user: str | None = None,
        timeout: float = 30.0,
        stdin: str | None = None,
    ) -> "ContainerExecResult":
        """Run ``command`` inside the container."""
        ...

    def write_file(
        self,
        handle: "ContainerHandle",
        path: str,
        content: str,
        *,
        user: str | None = None,
    ) -> bool:
        """Write ``content`` to ``path`` inside the container."""
        ...

    def read_file(
        self,
        handle: "ContainerHandle",
        path: str,
        *,
        user: str | None = None,
    ) -> tuple[str, bool]:
        """Read a file from inside the container. Returns (content, exists)."""
        ...

    def ensure_image(self, image: str) -> None:
        """Ensure the container image is available locally (pull if needed)."""
        ...

    def stop(self, handle: "ContainerHandle") -> None:
        """Stop and remove the container. Idempotent."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# LocalDockerRunner — reference implementation
# ─────────────────────────────────────────────────────────────────────────────


class LocalDockerRunner:
    """Container runner backed by the local ``docker`` CLI.

    One runner instance can manage many containers. Containers launched
    through this runner are tracked in a weak set and cleaned up on
    process exit via atexit.
    """

    _ATEXIT_REGISTERED = False
    _LIVE_HANDLES: "weakref.WeakSet[ContainerHandle]" = weakref.WeakSet()

    def __init__(self, docker_cmd: str = "docker") -> None:
        self._docker = docker_cmd
        if not LocalDockerRunner._ATEXIT_REGISTERED:
            atexit.register(LocalDockerRunner._atexit_cleanup)
            LocalDockerRunner._ATEXIT_REGISTERED = True

    # ── Lifecycle ────────────────────────────────────────────────────────

    def launch(self, spec: ContainerSpec) -> ContainerHandle:
        """Start a container from ``spec`` and return a handle.

        The container runs in detached mode with ``--rm`` so Docker
        auto-removes it on stop. We also track the handle and register
        an atexit hook so orphans are impossible.
        """
        name = f"{spec.name_prefix}-{uuid.uuid4().hex[:8]}"
        argv = [
            self._docker,
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            f"--network={spec.network}",
            f"--memory={spec.memory}",
            f"--cpus={spec.cpus}",
            f"--pids-limit={spec.pids_limit}",
            "-w",
            spec.workdir,
        ]
        for host_path, container_path, mode in spec.mounts:
            argv.extend(["-v", f"{host_path}:{container_path}:{mode}"])
        for key, value in spec.env.items():
            argv.extend(["-e", f"{key}={value}"])
        argv.append(spec.image)
        argv.extend(spec.command)

        result = self._run_docker(argv, timeout=30.0)
        if result.returncode != 0:
            raise ContainerRunnerError(f"docker run failed (exit={result.returncode}): {result.stderr.strip()}")
        container_id = result.stdout.strip()
        handle = ContainerHandle(
            container_id=container_id,
            name=name,
            spec=spec,
        )
        LocalDockerRunner._LIVE_HANDLES.add(handle)
        logger.info(
            "Container launched: %s (%s, image=%s, network=%s, mem=%s, cpus=%s)",
            name,
            container_id[:12],
            spec.image,
            spec.network,
            spec.memory,
            spec.cpus,
        )
        return handle

    def stop(self, handle: ContainerHandle) -> None:
        """Stop and remove the container. Idempotent."""
        result = self._run_docker(
            [self._docker, "rm", "-f", handle.container_id],
            timeout=15.0,
            check=False,
        )
        if result.returncode != 0:
            # Container may already be gone — log but don't raise.
            logger.debug(
                "docker rm -f %s returned %d: %s",
                handle.name,
                result.returncode,
                result.stderr.strip(),
            )
        try:
            LocalDockerRunner._LIVE_HANDLES.discard(handle)
        except Exception:
            pass
        logger.info("Container stopped: %s", handle.name)

    # ── Exec + file I/O ──────────────────────────────────────────────────

    def exec(
        self,
        handle: ContainerHandle,
        command: str,
        *,
        user: str | None = None,
        timeout: float = 30.0,
        stdin: str | None = None,
    ) -> ContainerExecResult:
        """Run ``command`` inside the container via ``docker exec``.

        Uses container-side ``timeout`` so runaway processes are killed
        inside the container, not just on the host side. Still wraps the
        host-side subprocess call in a slightly larger timeout as a
        safety net.
        """
        effective_user = user if user is not None else handle.spec.user
        # Container-side kill using the Linux `timeout` coreutil.
        # Shell varies by image (bash on Debian/Ubuntu/RHEL, sh on Alpine).
        container_cmd = [
            "timeout",
            "--signal=KILL",
            f"{timeout:.1f}",
            handle.spec.shell,
            "-c",
            command,
        ]
        argv = [self._docker, "exec"]
        if effective_user:
            argv.extend(["-u", effective_user])
        if stdin is not None:
            argv.append("-i")
        argv.append(handle.container_id)
        argv.extend(container_cmd)

        # Host-side timeout = container timeout + small cushion for
        # docker exec startup/teardown.
        host_timeout = timeout + 5.0
        result = self._run_docker(
            argv,
            timeout=host_timeout,
            stdin=stdin,
            check=False,
        )
        # timeout coreutil exits 124 on SIGTERM, 137 on SIGKILL
        timed_out = result.returncode in (124, 137)
        return ContainerExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            timed_out=timed_out,
        )

    def write_file(
        self,
        handle: ContainerHandle,
        path: str,
        content: str,
        *,
        user: str | None = "root",
    ) -> bool:
        """Write ``content`` to ``path`` inside the container.

        Defaults to ``user=root`` so honeypot files at paths like
        ``/etc/passwd`` can be populated at startup.
        """
        # mkdir -p the parent, then tee the content in.
        import os

        parent = os.path.dirname(path) or "/"
        mk_result = self.exec(
            handle,
            f"mkdir -p {parent!s}",
            user=user,
            timeout=5.0,
        )
        if mk_result.exit_code != 0:
            logger.warning(
                "mkdir %s failed in container %s: %s",
                parent,
                handle.name,
                mk_result.stderr,
            )
            return False
        # Use tee to write (handles arbitrary content via stdin).
        write_result = self.exec(
            handle,
            f"tee {path!s} > /dev/null",
            user=user,
            timeout=10.0,
            stdin=content,
        )
        return write_result.exit_code == 0

    def read_file(
        self,
        handle: ContainerHandle,
        path: str,
        *,
        user: str | None = None,
    ) -> tuple[str, bool]:
        """Read a file from inside the container. Returns (content, exists)."""
        result = self.exec(
            handle,
            f"cat {path!s} 2>/dev/null",
            user=user,
            timeout=10.0,
        )
        return result.stdout, result.exit_code == 0

    # ── Image management ─────────────────────────────────────────────────

    def ensure_image(self, image: str) -> None:
        """Pull ``image`` if not already present.

        Prints a one-line notice to stderr on pull to avoid silent
        long waits on first run.
        """
        inspect = self._run_docker(
            [self._docker, "image", "inspect", image],
            timeout=10.0,
            check=False,
        )
        if inspect.returncode == 0:
            return
        logger.info("Pulling Docker image: %s ...", image)
        pull = self._run_docker(
            [self._docker, "pull", image],
            timeout=300.0,
            check=False,
        )
        if pull.returncode != 0:
            raise ContainerRunnerError(f"docker pull {image} failed: {pull.stderr.strip()}")

    # ── Internals ────────────────────────────────────────────────────────

    def _run_docker(
        self,
        argv: Iterable[str],
        *,
        timeout: float,
        stdin: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Centralized docker CLI invocation (mockable in tests)."""
        try:
            result = subprocess.run(
                list(argv),
                capture_output=True,
                text=True,
                timeout=timeout,
                input=stdin,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            logger.warning("docker CLI timed out: %s", " ".join(list(argv)[:4]))
            return subprocess.CompletedProcess(
                args=list(argv),
                returncode=124,
                stdout=e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
                stderr="timeout",
            )
        except FileNotFoundError as e:
            raise ContainerRunnerError(f"docker CLI not found: {e}. Is Docker installed?") from e
        if check and result.returncode != 0:
            raise ContainerRunnerError(f"docker command failed: {' '.join(list(argv)[:4])}\n{result.stderr}")
        return result

    @classmethod
    def _atexit_cleanup(cls) -> None:
        """Force-remove any containers still tracked when the process exits."""
        for handle in list(cls._LIVE_HANDLES):
            try:
                subprocess.run(
                    ["docker", "rm", "-f", handle.container_id],
                    capture_output=True,
                    timeout=10.0,
                    check=False,
                )
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Availability check
# ─────────────────────────────────────────────────────────────────────────────


_DOCKER_AVAILABLE_CACHE: bool | None = None


def get_container_runner(backend: str = "local-docker") -> ContainerRunner:
    """Factory for container runners.

    Args:
        backend: one of ``"local-docker"`` (default), or future cloud
            runners (``"aws-fargate"``, ``"gcp-cloud-run"``,
            ``"azure-aci"``) once they're implemented.

    Raises:
        ValueError: if ``backend`` is unknown.
        ContainerRunnerError: if the chosen backend can't initialize
            (e.g. Docker daemon not reachable for ``"local-docker"``).
    """
    backend = (backend or "local-docker").lower()
    if backend in ("local-docker", "docker", "local"):
        return LocalDockerRunner()
    raise ValueError(
        f"Unknown container runner backend: {backend!r}. "
        f"Supported: 'local-docker'. Cloud runners "
        f"(aws-fargate, gcp-cloud-run, azure-aci) are not yet implemented."
    )


def check_docker_available(refresh: bool = False) -> bool:
    """Return True if the docker daemon is reachable.

    Result is cached on first call. Pass ``refresh=True`` to re-probe.
    """
    global _DOCKER_AVAILABLE_CACHE
    if _DOCKER_AVAILABLE_CACHE is not None and not refresh:
        return _DOCKER_AVAILABLE_CACHE
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5.0,
            check=False,
        )
        _DOCKER_AVAILABLE_CACHE = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        _DOCKER_AVAILABLE_CACHE = False
    return _DOCKER_AVAILABLE_CACHE
