# Docker Sandbox Plan

> **Status:** Not started. Independent — can be implemented at any time.
> **Integrates with:** Simulation Agent (primary consumer), Research Protocol (experiment isolation), normal agentic mode (safer tool execution)

## Vision

Replace the tmpdir-based simulation sandbox with Docker containers. Tool execution (bash, write_file, execute_file, etc.) happens inside an isolated container with a mounted workspace volume. The LLM and agent loop stay on the host — only side effects are sandboxed.

Falls back to tmpdir with an explicit user warning when Docker isn't available.

---

## Why Docker Over tmpdir

| Concern | tmpdir (current) | Docker (proposed) |
|---------|------------------|-------------------|
| **Filesystem isolation** | CWD-based, relies on Python path checks | True isolation — container has no access to host FS |
| **Process isolation** | None — bash runs as host user | Full — container runs as unprivileged user |
| **Network isolation** | None — AUT can curl anything | `--network=none` by default, opt-in access |
| **Resource limits** | None — runaway process can exhaust host | `--memory`, `--cpus`, `--pids-limit` |
| **Cleanup** | `shutil.rmtree` (can fail on open files) | `docker rm -f` (always works) |
| **Reproducibility** | Depends on host packages | Pinned image (e.g., `python:3.12-slim`) |
| **Security** | FearGatedExecutor is last line of defense | Docker is the last line — FearAgent is a bonus |

---

## Architecture

```
Host (Maxim process)
├── Orchestrator agent loop
├── AUT agent loop
├── LLMWorker (shared router, GPU access)
│
├── DockerSandbox
│   ├── Manages container lifecycle (create, exec, cleanup)
│   ├── Mounts /workspace volume from host tmpdir
│   └── Provides execute() interface matching current Executor
│
└── Tool execution flow:
    1. AUT LLM proposes: bash("ls -la")
    2. Executor calls DockerSandbox.execute("bash", "ls -la")
    3. DockerSandbox runs: docker exec <container> bash -c "ls -la"
    4. Captures stdout/stderr, exit code
    5. Returns ToolOutput to agent loop
```

### What runs WHERE

| Component | Where | Why |
|-----------|-------|-----|
| Agent loops | Host | Need access to LLM, memory, state |
| LLM inference | Host | Needs GPU / model files |
| SimulationBridge | Host | Thread-safe percept/action passing |
| `bash` tool | **Container** | Arbitrary shell commands |
| `write_file` tool | **Container** (via mounted volume) | File writes go to /workspace |
| `read_file` tool | **Container** (via mounted volume) | Reads from /workspace |
| `execute_file` tool | **Container** | Code execution |
| `respond` / `speak` | Host | No side effects, just returns text |
| `inspect_aut` | Host | Reads agent internals, no side effects |

---

## Core Component: `DockerSandbox`

```python
class DockerSandbox:
    """Manages a Docker container for sandboxed tool execution.

    Creates a container on init, provides execute() for running commands
    inside it, and cleans up on close. Falls back to tmpdir if Docker
    is unavailable.
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        workspace_dir: str | None = None,
        network: str = "none",
        memory_limit: str = "512m",
        cpu_limit: float = 1.0,
    ) -> None:
        self._image = image
        self._workspace = workspace_dir or tempfile.mkdtemp(prefix="maxim_sim_")
        self._network = network
        self._memory_limit = memory_limit
        self._cpu_limit = cpu_limit
        self._container_id: str | None = None

    def start(self) -> None:
        """Create and start the container with workspace mounted."""
        self._container_id = subprocess.check_output([
            "docker", "run", "-d",
            "--name", f"maxim-sandbox-{os.getpid()}",
            "-v", f"{self._workspace}:/workspace",
            "-w", "/workspace",
            f"--network={self._network}",
            f"--memory={self._memory_limit}",
            f"--cpus={self._cpu_limit}",
            "--pids-limit=50",
            "--user", "1000:1000",  # Unprivileged
            self._image,
            "sleep", "infinity",  # Keep alive for exec calls
        ], text=True).strip()

    def execute(self, command: str, timeout: float = 30.0) -> tuple[str, str, int]:
        """Run a command inside the container.

        Returns (stdout, stderr, exit_code).
        """
        result = subprocess.run(
            ["docker", "exec", self._container_id, "bash", "-c", command],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout, result.stderr, result.returncode

    def copy_to_workspace(self, host_path: str, container_path: str = ".") -> None:
        """Copy a file from host to the container's workspace."""
        dest = os.path.join(self._workspace, container_path)
        shutil.copy2(host_path, dest)

    def read_from_workspace(self, container_path: str) -> str:
        """Read a file from the container's workspace."""
        path = os.path.join(self._workspace, container_path)
        with open(path, "r") as f:
            return f.read()

    def cleanup(self) -> None:
        """Stop and remove the container + workspace."""
        if self._container_id:
            subprocess.run(
                ["docker", "rm", "-f", self._container_id],
                capture_output=True,
            )
        if self._workspace and os.path.isdir(self._workspace):
            shutil.rmtree(self._workspace, ignore_errors=True)
```

---

## Docker Availability Check + tmpdir Fallback

```python
def check_docker_available() -> bool:
    """Check if Docker daemon is running and accessible."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, timeout=5.0,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_sandbox(prefer_docker: bool = True) -> DockerSandbox | TmpdirSandbox:
    """Get the best available sandbox, with user confirmation for tmpdir.

    If Docker is available, uses it. If not, warns the user and asks
    for confirmation to use tmpdir fallback.
    """
    if prefer_docker and check_docker_available():
        return DockerSandbox()

    # Docker not available — warn and ask
    print()
    print("  ⚠  Docker is not available.")
    print("  Simulation sandboxing works best with Docker for full process")
    print("  and filesystem isolation. Without it, tools run in a temporary")
    print("  directory with limited protection.")
    print()
    print("  Install Docker: https://docs.docker.com/get-docker/")
    print()

    response = input("  Continue with tmpdir sandbox? (yes/no): ").strip().lower()
    if response not in ("yes", "y"):
        print("  Aborted. Install Docker and try again.")
        sys.exit(1)

    print("  Using tmpdir sandbox (reduced isolation)")
    return TmpdirSandbox()
```

---

## `TmpdirSandbox` (existing behavior, wrapped)

```python
class TmpdirSandbox:
    """Fallback sandbox using a temporary directory. No process isolation."""

    def __init__(self) -> None:
        self._workspace = tempfile.mkdtemp(prefix="maxim_sim_")

    def start(self) -> None:
        pass  # tmpdir already exists

    def execute(self, command: str, timeout: float = 30.0) -> tuple[str, str, int]:
        """Run command directly on host (in workspace dir)."""
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True, text=True, timeout=timeout,
            cwd=self._workspace,
        )
        return result.stdout, result.stderr, result.returncode

    def copy_to_workspace(self, host_path: str, container_path: str = ".") -> None:
        dest = os.path.join(self._workspace, container_path)
        shutil.copy2(host_path, dest)

    def read_from_workspace(self, container_path: str) -> str:
        path = os.path.join(self._workspace, container_path)
        with open(path, "r") as f:
            return f.read()

    def cleanup(self) -> None:
        if self._workspace and os.path.isdir(self._workspace):
            shutil.rmtree(self._workspace, ignore_errors=True)
```

Both implement the same interface. The orchestrator doesn't know or care which one is active.

---

## Integration Points

### Simulation Agent (primary consumer)

The orchestrator creates the sandbox during `start_simulation_mode()` and passes it to the AUT's executor. Tool calls go through `sandbox.execute()` instead of raw `subprocess.run()`.

```python
# In orchestrator.py:
sandbox = get_sandbox(prefer_docker=True)
sandbox.start()
# ... pass sandbox to AUT tool registry ...
# ... on cleanup:
sandbox.cleanup()
```

### Research Protocol (experiment isolation)

Each `spawn_sub_simulation` gets its own sandbox — experiments are fully isolated from each other. The Peer Reviewer's `run_validation_experiment` also gets a fresh sandbox, ensuring re-run independence.

### Normal Agentic Mode (optional)

The `--sandbox docker` CLI flag could enable Docker sandboxing for regular `--mode agentic` sessions. Not required — the current FearGatedExecutor + filesystem policy is adequate for non-adversarial use.

### Relation to Other Plans

| Plan | How Docker Sandbox Helps |
|------|--------------------------|
| **Simulation Agent** | Safer AUT execution, especially for adversarial testing |
| **Research Protocol** | Isolated experiment environments per spawn |
| **Multi-LLM Scaling** | Container could include local model server for sub-sim AUT |
| **Realtime Refinement** | Reproducible environments for consistent measurements |

**No blockers.** Can be implemented at any time. Doesn't depend on any other plan.

---

## Implementation Plan

### Phase 1: DockerSandbox + TmpdirSandbox (~200 LOC)

1. `simulation/sandbox.py` — `DockerSandbox`, `TmpdirSandbox`, `check_docker_available()`, `get_sandbox()`
2. Common interface: `start()`, `execute()`, `copy_to_workspace()`, `read_from_workspace()`, `cleanup()`
3. Docker availability check with user-facing warning and install link

### Phase 2: Wire into Simulation (~100 LOC)

1. `orchestrator.py` calls `get_sandbox()` during startup
2. AUT's `BashTool` and `ExecuteFileTool` route through `sandbox.execute()`
3. `write_file` / `read_file` use `sandbox.copy_to_workspace()` / `read_from_workspace()`
4. Cleanup in finally block (both Docker and tmpdir)

### Phase 2.5: LLM Resource Planning (~100 LOC)

Pre-spawn LLM call to determine container specs:

1. `plan_container_resources()` — queries system resources (psutil / /proc), formats prompt
2. LLM returns: image, memory limit, CPU limit, network mode, pre-install packages
3. Fallback to defaults if LLM call fails or Docker unavailable
4. Resource spec displayed to user before container creation

This runs once per simulation start. The spec is reused for all `spawn_sub_simulation` calls in that session (sub-sims inherit the parent spec unless the goal suggests different requirements).

### Phase 3: CLI + Config (~50 LOC)

1. `--sandbox docker|tmpdir|auto` CLI flag (default: `auto` = prefer Docker)
2. `--sandbox-image` for custom Docker image (default: `python:3.12-slim`)
3. `--sandbox-network` for network mode (default: `none`)

### Phase 4: Tests (~100 LOC)

1. Test `check_docker_available()` (mock subprocess)
2. Test `TmpdirSandbox` execute/copy/read/cleanup
3. Test `DockerSandbox` execute/copy/read/cleanup (marked `@pytest.mark.slow`, requires Docker)
4. Test fallback flow (Docker unavailable → user prompt)

**Total: ~550 LOC**

---

## Design Decisions

**Q: Why `sleep infinity` + `docker exec` instead of `docker run` per command?**
A: Container startup takes 1-2s. With `sleep infinity`, the container stays warm and `docker exec` adds only ~50ms overhead per command. For a simulation with 10+ tool calls, this saves 10-20s.

**Q: Why mount a host tmpdir instead of using Docker volumes?**
A: Host tmpdir mount lets us read/write files from both sides without extra copy commands. The orchestrator can inspect workspace contents directly (for the simulation report) without `docker cp`.

**Q: What Docker image should we use?**
A: `python:3.12-slim` by default — matches the host Python version, includes pip, has bash. Users can override with `--sandbox-image` for custom environments (e.g., `node:20-slim` for JS testing).

**Q: What about Docker-in-Docker for nested containers?**
A: Not needed. The AUT runs commands, not containers. If a future use case needs it, add `--privileged` as an opt-in flag.

**Q: Can the AUT install packages inside the container?**
A: Yes — `pip install` works inside the container. Installed packages persist for the session (container stays alive). Each `spawn_sub_simulation` gets a fresh container, so installs don't leak between sub-sims.

**Q: How are container resources allocated?**
A: An LLM planning call before container creation. The orchestrator provides: available system resources (CPU cores, free RAM, GPU VRAM, disk), the experiment goal, and the AUT's tool requirements. The LLM returns a resource spec:

```python
# Pre-spawn LLM call:
resource_spec = llm_router.generate_json(
    prompt=f"""Given these available resources:
    - CPU: {available_cpus} cores ({cpu_usage_pct}% used)
    - RAM: {free_ram_gb}GB free of {total_ram_gb}GB
    - Disk: {free_disk_gb}GB free
    - GPU: {gpu_info}

    And this experiment: {goal}
    With these tools needed: {tool_list}

    Return a Docker resource spec as JSON:
    {{"image": "...", "memory": "...", "cpus": ..., "network": "none|bridge", "packages": [...]}}
    """,
    system_override="You are a DevOps assistant. Return ONLY valid JSON.",
)
```

This lets the system adapt: a bash-only safety test gets `python:3.12-slim` with 256MB, while a code execution test with ML dependencies gets a larger image with 2GB. The fallback if the LLM call fails is the default spec (`python:3.12-slim`, 512MB, 1 CPU).

**Q: What happens if Docker is available but the image isn't pulled?**
A: `docker run` auto-pulls the image on first use. This adds a one-time download (30-50MB for slim images). The first `start()` call will be slow; subsequent calls use the cached image.
