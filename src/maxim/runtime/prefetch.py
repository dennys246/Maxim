"""Speculative pre-fetching for efficient LLM context building.

This module provides:
1. File reference detection from user input
2. Speculative pre-fetching of likely needed files
3. Result caching with TTL and file mtime validation
4. New file vs modify detection for skipping exploration
5. Systematic file discovery using find/glob patterns

The goal is to reduce LLM calls by pre-gathering context before the LLM
even needs to request it. With aggressive pre-fetching, file modifications
can often be done in 1 LLM call instead of 2+.

Workflow:
  User mentions file → Auto-discover with find/glob → Read file + related files
  → LLM gets full context in first call → LLM can write immediately
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# File Reference Detection
# ─────────────────────────────────────────────────────────────────────────────

# Patterns to detect file references in user input (pre-compiled for performance)
FILE_PATTERNS = [
    # Explicit file extensions
    re.compile(r"(\b\w+\.py\b)", re.IGNORECASE),  # Python files
    re.compile(r"(\b\w+\.js\b)", re.IGNORECASE),  # JavaScript
    re.compile(r"(\b\w+\.ts\b)", re.IGNORECASE),  # TypeScript
    re.compile(r"(\b\w+\.json\b)", re.IGNORECASE),  # JSON
    re.compile(r"(\b\w+\.yaml\b)", re.IGNORECASE),  # YAML
    re.compile(r"(\b\w+\.yml\b)", re.IGNORECASE),  # YAML alt
    re.compile(r"(\b\w+\.md\b)", re.IGNORECASE),  # Markdown
    re.compile(r"(\b\w+\.txt\b)", re.IGNORECASE),  # Text
    re.compile(r"(\b\w+\.sh\b)", re.IGNORECASE),  # Shell scripts
    re.compile(r"(\b\w+\.css\b)", re.IGNORECASE),  # CSS
    re.compile(r"(\b\w+\.html\b)", re.IGNORECASE),  # HTML
    # Path patterns
    re.compile(r"(src/[\w/]+\.?\w*)", re.IGNORECASE),  # src/ paths
    re.compile(r"(lib/[\w/]+\.?\w*)", re.IGNORECASE),  # lib/ paths
    re.compile(r"(tests?/[\w/]+\.?\w*)", re.IGNORECASE),  # test(s)/ paths
    re.compile(r"(config/[\w/]+\.?\w*)", re.IGNORECASE),  # config/ paths
]

# Keywords indicating file modification intent
MODIFY_KEYWORDS = frozenset({
    "update", "modify", "change", "edit", "fix", "add to", "append",
    "remove from", "delete from", "refactor", "improve", "enhance",
})

# Keywords indicating new file creation
CREATE_KEYWORDS = frozenset({
    "create", "new", "make", "write", "generate", "build",
})


@dataclass
class FileReference:
    """A detected file reference from user input."""

    pattern: str  # The matched pattern (e.g., "hello.py")
    is_path: bool  # Whether it looks like a full path
    intent: str  # "modify", "create", or "unknown"
    confidence: float  # How confident we are in this detection


def detect_file_references(text: str) -> list[FileReference]:
    """Detect file references in user input.

    Returns list of FileReference objects sorted by confidence.
    """
    refs: list[FileReference] = []
    text_lower = text.lower()

    # Detect modification vs creation intent
    has_modify_intent = any(kw in text_lower for kw in MODIFY_KEYWORDS)
    has_create_intent = any(kw in text_lower for kw in CREATE_KEYWORDS)

    if has_create_intent and not has_modify_intent:
        intent = "create"
    elif has_modify_intent:
        intent = "modify"
    else:
        intent = "unknown"

    seen = set()
    for compiled_pattern in FILE_PATTERNS:
        for match in compiled_pattern.finditer(text):  # Use pre-compiled pattern
            file_ref = match.group(1)
            if file_ref.lower() in seen:
                continue
            seen.add(file_ref.lower())

            is_path = "/" in file_ref
            confidence = 0.9 if is_path else 0.7

            refs.append(FileReference(
                pattern=file_ref,
                is_path=is_path,
                intent=intent,
                confidence=confidence,
            ))

    # Sort by confidence descending
    refs.sort(key=lambda r: -r.confidence)
    return refs


def detect_file_intent(text: str) -> str:
    """Detect whether user wants to create or modify a file.

    Returns: "create", "modify", or "unknown"
    """
    text_lower = text.lower()

    has_modify = any(kw in text_lower for kw in MODIFY_KEYWORDS)
    has_create = any(kw in text_lower for kw in CREATE_KEYWORDS)

    if has_create and not has_modify:
        return "create"
    elif has_modify:
        return "modify"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Systematic File Discovery (using bash find)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiscoveredFile:
    """A file discovered during systematic search."""

    path: str
    name: str
    size: int
    content: str | None = None
    is_main_target: bool = False  # True if this is the primary file user mentioned
    is_test_file: bool = False
    is_config_file: bool = False


def discover_files_with_find(
    filename: str,
    base_path: str = ".",
    max_files: int = 10,
    max_content_size: int = 10000,
    include_related: bool = True,
) -> list[DiscoveredFile]:
    """Use bash find to discover files matching a pattern.

    This is more powerful than glob for complex patterns and can find
    files across the entire project quickly.

    Args:
        filename: The filename pattern to search for (e.g., "hello.py")
        base_path: Base directory to search from
        max_files: Maximum number of files to return
        max_content_size: Maximum bytes to read from each file
        include_related: Whether to also find test files, configs, etc.

    Returns:
        List of DiscoveredFile objects with paths and optionally content
    """
    discovered: list[DiscoveredFile] = []

    # Extract base name without extension for related file search
    base_name = Path(filename).stem
    extension = Path(filename).suffix

    # Build find patterns
    patterns = [
        f"*{filename}*",  # Main file pattern
    ]

    if include_related and base_name:
        # Add test file patterns
        patterns.extend([
            f"*test_{base_name}*",
            f"*{base_name}_test*",
            f"*{base_name}Test*",
        ])
        # Add config patterns if it looks like a source file
        if extension in (".py", ".js", ".ts", ".go", ".rs"):
            patterns.extend([
                "*config*.py",
                "*config*.json",
                "*settings*.py",
                "*settings*.json",
            ])

    # Directories to exclude from search (virtual envs, caches, vendor dirs)
    exclude_dirs = [".venv", "venv", "node_modules", "__pycache__", ".git", ".tox", "dist", "build", "site-packages"]

    try:
        # Run find command for each pattern (limited for safety)
        all_found: list[str] = []
        for pattern in patterns[:5]:  # Limit patterns
            try:
                # Build find command with exclusions
                find_cmd = ["find", base_path, "-maxdepth", "6"]
                # Add exclusion clauses for each excluded directory
                for exclude_dir in exclude_dirs:
                    find_cmd.extend(["-path", f"*/{exclude_dir}/*", "-prune", "-o"])
                # Add the actual search criteria
                find_cmd.extend(["-name", pattern, "-type", "f", "-print"])

                result = subprocess.run(
                    find_cmd,
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                if result.returncode == 0 and result.stdout.strip():
                    found_files = result.stdout.strip().split("\n")
                    # Filter out any empty lines or pruned paths
                    found_files = [f for f in found_files if f and not any(excl in f for excl in exclude_dirs)]
                    all_found.extend(found_files[:max_files])
            except subprocess.TimeoutExpired:
                logger.debug("Find command timed out for pattern: %s", pattern)
            except Exception as e:
                logger.debug("Find command failed: %s", e)

        # Deduplicate and limit
        seen_paths = set()
        unique_files: list[str] = []
        for path in all_found:
            real_path = os.path.realpath(path)
            if real_path not in seen_paths:
                seen_paths.add(real_path)
                unique_files.append(path)
                if len(unique_files) >= max_files:
                    break

        # Read file contents and classify
        for file_path in unique_files:
            try:
                stat = os.stat(file_path)
                file_name = os.path.basename(file_path)

                # Determine file type
                is_main = filename.lower() in file_name.lower()
                is_test = "test" in file_name.lower()
                is_config = "config" in file_name.lower() or "settings" in file_name.lower()

                # Read content if file is small enough
                content = None
                if stat.st_size <= max_content_size:
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read(max_content_size)
                    except Exception:
                        pass

                discovered.append(DiscoveredFile(
                    path=file_path,
                    name=file_name,
                    size=stat.st_size,
                    content=content,
                    is_main_target=is_main,
                    is_test_file=is_test,
                    is_config_file=is_config,
                ))

            except Exception as e:
                logger.debug("Failed to process discovered file %s: %s", file_path, e)

        # Sort: main target first, then tests, then configs, then others
        discovered.sort(key=lambda f: (
            not f.is_main_target,
            not f.is_test_file,
            not f.is_config_file,
            f.path,
        ))

    except Exception as e:
        logger.debug("File discovery failed: %s", e)

    return discovered


def discover_and_read_files(
    file_refs: list[FileReference],
    base_path: str = ".",
    max_total_content: int = 50000,
) -> dict[str, str]:
    """Discover and read all referenced files and their related files.

    This is the main entry point for systematic pre-fetching. Given a list
    of file references from user input, it:
    1. Uses find to locate each file
    2. Reads the file content
    3. Finds and reads related files (tests, configs)
    4. Returns a dict of path -> content

    Args:
        file_refs: List of FileReference from detect_file_references
        base_path: Base directory to search from
        max_total_content: Maximum total bytes to read across all files

    Returns:
        Dict mapping file paths to their content
    """
    contents: dict[str, str] = {}
    total_size = 0

    for ref in file_refs[:3]:  # Limit to first 3 file references
        if total_size >= max_total_content:
            break

        # Discover files matching this reference
        discovered = discover_files_with_find(
            filename=ref.pattern,
            base_path=base_path,
            max_files=8,
            max_content_size=min(15000, max_total_content - total_size),
            include_related=(ref.intent == "modify"),
        )

        for df in discovered:
            if df.content and df.path not in contents:
                contents[df.path] = df.content
                total_size += len(df.content)
                if total_size >= max_total_content:
                    break

    return contents


# ─────────────────────────────────────────────────────────────────────────────
# Result Caching
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    """A cached result with TTL and validation."""

    result: Any
    timestamp: float
    file_mtime: float | None = None  # For file reads, track mtime
    ttl_seconds: float = 30.0


class ResultCache:
    """Cache for tool results with TTL and file mtime validation."""

    def __init__(self, default_ttl: float = 30.0, max_entries: int = 100):
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl
        self._max_entries = max_entries

    def _make_key(self, tool_name: str, params: dict[str, Any]) -> str:
        """Create a cache key from tool name and params."""
        # Sort params for consistent keys
        sorted_params = sorted(params.items())
        param_str = str(sorted_params)
        return f"{tool_name}:{hashlib.md5(param_str.encode()).hexdigest()}"

    def get(self, tool_name: str, params: dict[str, Any]) -> Any | None:
        """Get a cached result if valid.

        Returns None if not cached or expired.
        """
        key = self._make_key(tool_name, params)

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            # Check TTL
            age = time.time() - entry.timestamp
            if age > entry.ttl_seconds:
                del self._cache[key]
                return None

            # For file reads, check mtime
            if entry.file_mtime is not None:
                path = params.get("path")
                if path:
                    try:
                        current_mtime = os.path.getmtime(path)
                        if current_mtime != entry.file_mtime:
                            del self._cache[key]
                            return None
                    except OSError:
                        del self._cache[key]
                        return None

            return entry.result

    def put(
        self,
        tool_name: str,
        params: dict[str, Any],
        result: Any,
        ttl: float | None = None,
    ) -> None:
        """Cache a result."""
        key = self._make_key(tool_name, params)

        # Get file mtime if this is a file read
        file_mtime = None
        if tool_name == "read_file":
            path = params.get("path")
            if path:
                try:
                    file_mtime = os.path.getmtime(path)
                except OSError:
                    pass

        with self._lock:
            # Evict old entries if at capacity
            if len(self._cache) >= self._max_entries:
                # Remove oldest entries
                sorted_entries = sorted(
                    self._cache.items(),
                    key=lambda x: x[1].timestamp,
                )
                for old_key, _ in sorted_entries[:self._max_entries // 4]:
                    del self._cache[old_key]

            self._cache[key] = CacheEntry(
                result=result,
                timestamp=time.time(),
                file_mtime=file_mtime,
                ttl_seconds=ttl or self._default_ttl,
            )

    def invalidate(self, tool_name: str | None = None, path: str | None = None) -> int:
        """Invalidate cache entries.

        Args:
            tool_name: If provided, invalidate entries for this tool
            path: If provided, invalidate entries containing this path

        Returns: Number of entries invalidated
        """
        with self._lock:
            to_remove = []
            for key, entry in self._cache.items():
                if tool_name and key.startswith(f"{tool_name}:"):
                    to_remove.append(key)
                elif path:
                    # Check if result contains this path
                    result_str = str(entry.result)
                    if path in result_str:
                        to_remove.append(key)

            for key in to_remove:
                del self._cache[key]

            return len(to_remove)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "max_entries": self._max_entries,
                "default_ttl": self._default_ttl,
            }


# Global cache instance
_result_cache = ResultCache()


def get_result_cache() -> ResultCache:
    """Get the global result cache."""
    return _result_cache


# ─────────────────────────────────────────────────────────────────────────────
# Speculative Pre-fetcher
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PrefetchResult:
    """Results from speculative pre-fetching."""

    file_references: list[FileReference]
    intent: str  # "create", "modify", "unknown"
    prefetched_context: dict[str, Any] = field(default_factory=dict)
    glob_results: list[dict[str, Any]] = field(default_factory=list)
    file_contents: dict[str, str] = field(default_factory=dict)
    related_files: list[str] = field(default_factory=list)
    skip_exploration: bool = False  # True if we can skip LLM exploration call


class SpeculativePrefetcher:
    """Pre-fetches likely needed context based on user input.

    When a user mentions a file, this class:
    1. Detects file references and intent (create vs modify)
    2. Speculatively runs glob to find the file
    3. Reads the file and related files (tests, configs)
    4. Caches results for immediate use

    This allows the LLM to receive pre-gathered context and potentially
    skip the exploration call entirely.
    """

    def __init__(
        self,
        executor: Any | None = None,
        cache: ResultCache | None = None,
        base_path: str = ".",
    ):
        self._executor = executor
        self._cache = cache or get_result_cache()
        self._base_path = base_path

    def set_executor(self, executor: Any) -> None:
        """Set the tool executor (for deferred initialization)."""
        self._executor = executor

    def prefetch_for_input(self, user_input: str) -> PrefetchResult:
        """Analyze user input and pre-fetch likely needed context.

        Args:
            user_input: The user's input text

        Returns:
            PrefetchResult with pre-gathered context
        """
        result = PrefetchResult(
            file_references=[],
            intent="unknown",
        )

        # Detect file references
        refs = detect_file_references(user_input)
        result.file_references = refs
        result.intent = detect_file_intent(user_input)

        if not refs:
            return result

        # If creating a new file and we detect a clear filename, skip exploration
        if result.intent == "create" and len(refs) == 1:
            ref = refs[0]
            # Check if file already exists
            potential_path = self._find_file(ref.pattern)
            if potential_path is None:
                # File doesn't exist - this is truly a new file
                result.skip_exploration = True
                logger.info("Skipping exploration - new file creation: %s", ref.pattern)
                return result

        # For modifications, use systematic discovery with find
        # This is more aggressive - we find and read files BEFORE the LLM runs
        if result.intent in ("modify", "unknown"):
            try:
                # Use the new systematic file discovery
                discovered_contents = discover_and_read_files(
                    file_refs=refs,
                    base_path=self._base_path,
                    max_total_content=50000,
                )
                result.file_contents.update(discovered_contents)

                # If we found the main file and have its content, we can skip exploration
                if discovered_contents:
                    main_file_found = any(
                        ref.pattern.lower() in path.lower()
                        for ref in refs
                        for path in discovered_contents.keys()
                    )
                    if main_file_found and len(discovered_contents) >= 1:
                        result.skip_exploration = True
                        logger.info(
                            "Systematic discovery found %d files - LLM can write directly",
                            len(discovered_contents),
                        )

            except Exception as e:
                logger.debug("Systematic discovery failed, falling back: %s", e)
                # Fall back to old method if discovery fails
                if self._executor:
                    self._prefetch_file_context(refs, result)

        return result

    def _find_file(self, pattern: str) -> str | None:
        """Try to find a file matching the pattern."""
        # Try direct path first
        if os.path.isfile(pattern):
            return pattern

        # Try with base path
        full_path = os.path.join(self._base_path, pattern)
        if os.path.isfile(full_path):
            return full_path

        # Try glob search
        if self._executor:
            try:
                glob_result = self._execute_cached(
                    "glob",
                    {"pattern": f"**/{pattern}", "max_results": 1},
                )
                if glob_result and glob_result.get("matches"):
                    return glob_result["matches"][0].get("path")
            except Exception as e:
                logger.debug("Glob search failed: %s", e)

        return None

    def _prefetch_file_context(
        self,
        refs: list[FileReference],
        result: PrefetchResult,
    ) -> None:
        """Pre-fetch file contents and related files."""
        for ref in refs[:3]:  # Limit to first 3 references
            # Find the file
            file_path = self._find_file(ref.pattern)

            if file_path:
                # Read the file
                try:
                    content = self._execute_cached(
                        "read_file",
                        {"path": file_path},
                    )
                    if content:
                        result.file_contents[file_path] = str(content)[:5000]
                except Exception as e:
                    logger.debug("Failed to read %s: %s", file_path, e)

                # Find related files (tests, configs)
                self._prefetch_related_files(file_path, result)
            else:
                # File not found - try glob to find similar files
                try:
                    glob_result = self._execute_cached(
                        "glob",
                        {"pattern": f"**/*{ref.pattern}*", "max_results": 10},
                    )
                    if glob_result:
                        result.glob_results.append(glob_result)
                except Exception as e:
                    logger.debug("Glob failed for %s: %s", ref.pattern, e)

    def _prefetch_related_files(
        self,
        file_path: str,
        result: PrefetchResult,
    ) -> None:
        """Prefetch related files (tests, configs, similar files)."""
        path = Path(file_path)
        filename = path.stem
        extension = path.suffix

        # Common related file patterns
        related_patterns = [
            f"**/test_{filename}{extension}",  # test_foo.py
            f"**/{filename}_test{extension}",  # foo_test.py
            f"**/tests/test_{filename}{extension}",  # tests/test_foo.py
        ]

        # Add config files if this is a source file
        if extension in (".py", ".js", ".ts"):
            related_patterns.extend([
                "**/config*.py",
                "**/config*.json",
                "**/settings*.py",
            ])

        for pattern in related_patterns[:4]:  # Limit patterns
            try:
                glob_result = self._execute_cached(
                    "glob",
                    {"pattern": pattern, "max_results": 3},
                )
                if glob_result and glob_result.get("matches"):
                    for match in glob_result["matches"][:2]:
                        related_path = match.get("path")
                        if related_path and related_path not in result.file_contents:
                            result.related_files.append(related_path)
                            # Read the related file
                            try:
                                content = self._execute_cached(
                                    "read_file",
                                    {"path": related_path},
                                )
                                if content:
                                    result.file_contents[related_path] = str(content)[:3000]
                            except Exception:
                                pass
            except Exception as e:
                logger.debug("Related file search failed: %s", e)

    def _execute_cached(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Execute a tool with caching."""
        # Check cache first
        cached = self._cache.get(tool_name, params)
        if cached is not None:
            logger.debug("Cache hit for %s", tool_name)
            return cached

        # Execute tool
        if not self._executor:
            return None

        try:
            action = {"tool_name": tool_name, "params": params}
            result = self._executor.execute(action)

            if getattr(result, "success", False):
                output = getattr(result, "output", None)
                # Cache the result
                self._cache.put(tool_name, params, output)
                return output
        except Exception as e:
            logger.debug("Tool execution failed: %s", e)

        return None

    def format_prefetch_context(
        self,
        result: PrefetchResult,
        max_total_chars: int = 3000,  # ~750 tokens, conservative for small context windows
    ) -> str:
        """Format prefetched results for LLM context.

        Returns a formatted string suitable for including in the LLM prompt.
        Limits total size to avoid exceeding context window.
        """
        if not result.file_contents and not result.glob_results:
            return ""

        parts = ["=" * 60]
        parts.append(">>> AUTOMATIC FILE DISCOVERY COMPLETE <<<")
        parts.append("=" * 60)
        parts.append(f"Intent detected: {result.intent}")
        parts.append(f"Files discovered: {len(result.file_contents)}")

        if result.file_references:
            parts.append(f"User mentioned: {', '.join(r.pattern for r in result.file_references)}")

        if result.skip_exploration:
            parts.append("")
            parts.append("*** EXPLORATION ALREADY DONE - YOU CAN PROCEED DIRECTLY ***")
            parts.append("All relevant files have been found and read. You can now:")
            parts.append("- For MODIFICATION: Use write_file with overwrite=True")
            parts.append("- For NEW FILE: Use write_file (no overwrite needed)")
            parts.append("- You do NOT need to use glob or read_file first!")
            parts.append("")

        if result.file_contents:
            parts.append("\n" + "-" * 40)
            parts.append("FILE CONTENTS (already read for you):")
            parts.append("-" * 40)

            # Classify files
            main_files = []
            test_files = []
            config_files = []
            other_files = []

            for path, content in result.file_contents.items():
                name = os.path.basename(path).lower()
                if any(ref.pattern.lower() in path.lower() for ref in result.file_references):
                    main_files.append((path, content))
                elif "test" in name:
                    test_files.append((path, content))
                elif "config" in name or "settings" in name:
                    config_files.append((path, content))
                else:
                    other_files.append((path, content))

            # Track remaining space for content
            current_len = sum(len(p) for p in parts)
            remaining_chars = max_total_chars - current_len - 200  # Reserve 200 for footer

            # Output in priority order: main files first (they get more space)
            # Main target files are most important, then tests, configs, others
            for label, files, max_per_file in [
                ("TARGET FILE", main_files, remaining_chars // max(1, len(main_files)) if main_files else 3000),
                ("TEST FILES", test_files, 1000),  # Limit test files to 1000 chars each
                ("CONFIG FILES", config_files, 500),  # Limit config files
                ("RELATED FILES", other_files, 500),
            ]:
                for path, content in files:
                    if remaining_chars <= 100:
                        parts.append(f"\n[{label}] {path}: (skipped - context limit)")
                        continue

                    parts.append(f"\n[{label}] {path}:")
                    if len(content) > max_per_file:
                        truncated = content[:max_per_file]
                        # Try to truncate at a line boundary
                        last_newline = truncated.rfind("\n")
                        if last_newline > max_per_file // 2:
                            truncated = truncated[:last_newline]
                        parts.append(truncated)
                        parts.append(f"\n... (truncated, {len(content) - len(truncated)} more chars)")
                        remaining_chars -= len(truncated) + 50
                    else:
                        parts.append(content)
                        remaining_chars -= len(content)

        if result.related_files:
            parts.append(f"\nAdditional files found (not read): {', '.join(result.related_files)}")

        if result.glob_results:
            parts.append("\n--- Directory Search Results ---")
            for gr in result.glob_results:
                if gr.get("matches"):
                    matches = [m.get("path", "?") for m in gr["matches"][:5]]
                    parts.append(f"Pattern '{gr.get('pattern', '?')}': {', '.join(matches)}")

        parts.append("\n" + "=" * 60)
        parts.append(">>> END FILE DISCOVERY <<<")
        parts.append("=" * 60)

        return "\n".join(parts)


# Global prefetcher instance
_prefetcher: SpeculativePrefetcher | None = None


def get_prefetcher() -> SpeculativePrefetcher:
    """Get the global prefetcher instance."""
    global _prefetcher
    if _prefetcher is None:
        _prefetcher = SpeculativePrefetcher()
    return _prefetcher


def init_prefetcher(executor: Any, base_path: str = ".") -> SpeculativePrefetcher:
    """Initialize the global prefetcher with an executor."""
    global _prefetcher
    _prefetcher = SpeculativePrefetcher(
        executor=executor,
        base_path=base_path,
    )
    return _prefetcher


# ─────────────────────────────────────────────────────────────────────────────
# Response Templates
# ─────────────────────────────────────────────────────────────────────────────

RESPONSE_TEMPLATES = {
    "file_created": "Created {filename} successfully ({lines} lines, {bytes} bytes).",
    "file_updated": "Updated {filename} - {summary}",
    "file_not_found": "Could not find file matching '{pattern}'. Would you like me to create it?",
    "exploration_complete": "Found {count} files matching your request. Ready to proceed with modifications.",
    "write_complete": "File operation complete: {filename}",
}


def get_response_template(template_name: str) -> str | None:
    """Get a response template by name."""
    return RESPONSE_TEMPLATES.get(template_name)


def format_response_template(template_name: str, **kwargs: Any) -> str | None:
    """Format a response template with the given values."""
    template = RESPONSE_TEMPLATES.get(template_name)
    if template:
        try:
            return template.format(**kwargs)
        except KeyError:
            return template
    return None