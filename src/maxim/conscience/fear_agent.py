"""FearAgent: General-purpose safety enforcement layer.

Provides risk assessment for potentially dangerous actions before execution.
Uses local LLM for code/action analysis with deterministic settings.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.models.language.router import LLMRouter

logger = logging.getLogger(__name__)


class DangerCategory(Enum):
    """Categories of potentially dangerous behavior."""

    CODE_EXECUTION = "code_execution"
    NETWORK_ACCESS = "network_access"
    FILE_SYSTEM = "file_system"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    PERSISTENCE = "persistence"
    OBFUSCATION = "obfuscation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    UNKNOWN = "unknown"


class RiskLevel(Enum):
    """Risk assessment levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Finding:
    """A single security finding."""

    category: DangerCategory
    description: str
    location: str = ""  # file:line or action identifier
    severity: RiskLevel = RiskLevel.MEDIUM
    evidence: str = ""  # code snippet or action details


@dataclass
class ReviewResult:
    """Result of a FearAgent review."""

    allow: bool
    risk: RiskLevel
    findings: list[Finding] = field(default_factory=list)
    summary: str = ""
    reviewed_at: str = ""  # ISO timestamp
    reviewer: str = "FearAgent"


class FearAgent:
    """General-purpose safety enforcement agent.

    FearAgent reviews actions and code before execution, providing
    risk assessment and blocking dangerous operations when necessary.
    """

    # Patterns that trigger automatic review
    SUSPICIOUS_PATTERNS: dict[DangerCategory, list[str]] = {
        DangerCategory.CODE_EXECUTION: [
            r"\bsubprocess\b",
            r"\bos\.system\b",
            r"\bos\.popen\b",
            r"\beval\s*\(",
            r"\bexec\s*\(",
            r"\bcompile\s*\(",
            r"__import__\s*\(",
            r"\bimportlib\b",
        ],
        DangerCategory.NETWORK_ACCESS: [
            r"\bsocket\b",
            r"\brequests\b",
            r"\burllib\b",
            r"\bhttpx\b",
            r"\baiohttp\b",
            r"\bftplib\b",
            r"\bsmtplib\b",
        ],
        DangerCategory.FILE_SYSTEM: [
            r"\.write\s*\(",
            r"\.unlink\s*\(",
            r"\bos\.remove\b",
            r"\bshutil\.rmtree\b",
            r"\bos\.chmod\b",
            r"\bos\.chown\b",
        ],
        DangerCategory.OBFUSCATION: [
            r"base64\.b64decode.*exec",
            r"zlib\.decompress.*exec",
            r"\bmarshal\.loads\b",
            r"\bpickle\.loads\b",
        ],
        DangerCategory.PERSISTENCE: [
            r"\bcrontab\b",
            r"systemd",
            r"\.bashrc",
            r"\.profile",
            r"autostart",
            r"HKEY_",
        ],
    }

    def __init__(self, llm: LLMRouter | None = None) -> None:
        """Initialize FearAgent.

        Args:
            llm: Optional LLM router for code analysis. If not provided,
                 FearAgent will use pattern matching only (less accurate).
        """
        self._llm = llm
        self._enabled = True
        self._strict_mode = False  # If True, block on any finding

    @property
    def enabled(self) -> bool:
        """Check if FearAgent is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable FearAgent."""
        self._enabled = bool(value)

    @property
    def strict_mode(self) -> bool:
        """Check if strict mode is enabled."""
        return self._strict_mode

    @strict_mode.setter
    def strict_mode(self, value: bool) -> None:
        """Enable or disable strict mode."""
        self._strict_mode = bool(value)

    def review_code(
        self,
        code: str,
        *,
        source: str = "unknown",
        context: str = "",
    ) -> ReviewResult:
        """Review code for security risks.

        Args:
            code: The code to review.
            source: Source identifier (e.g., "huggingface:model_id").
            context: Additional context about why this code is being loaded.

        Returns:
            ReviewResult with allow/deny decision and findings.
        """
        if not self._enabled:
            return ReviewResult(
                allow=True,
                risk=RiskLevel.LOW,
                summary="FearAgent disabled",
                reviewed_at=datetime.utcnow().isoformat(),
            )

        findings: list[Finding] = []

        # Pattern-based scanning (fast, always runs)
        for category, patterns in self.SUSPICIOUS_PATTERNS.items():
            for pattern in patterns:
                try:
                    matches = list(re.finditer(pattern, code, re.IGNORECASE))
                    for match in matches:
                        # Find line number
                        line_num = code[: match.start()].count("\n") + 1
                        findings.append(
                            Finding(
                                category=category,
                                description=f"Suspicious pattern: {pattern}",
                                location=f"{source}:{line_num}",
                                severity=RiskLevel.MEDIUM,
                                evidence=code[
                                    max(0, match.start() - 50) : match.end() + 50
                                ],
                            )
                        )
                except re.error:
                    # Skip invalid patterns
                    continue

        # LLM-based analysis (more accurate, if available)
        if self._llm and len(code) < 50000:  # Limit size for LLM
            llm_findings = self._analyze_with_llm(code, source, context)
            findings.extend(llm_findings)

        # Determine overall risk and allow decision
        risk = self._calculate_risk(findings)
        allow = self._should_allow(findings, risk)

        return ReviewResult(
            allow=allow,
            risk=risk,
            findings=findings,
            summary=self._summarize_findings(findings),
            reviewed_at=datetime.utcnow().isoformat(),
        )

    def review_action(
        self,
        action_type: str,
        action_params: dict[str, Any],
        *,
        agent_id: str = "",
    ) -> ReviewResult:
        """Review an agent action before execution.

        Args:
            action_type: Type of action (e.g., "file_write", "shell_exec").
            action_params: Parameters for the action.
            agent_id: Identifier of the requesting agent.

        Returns:
            ReviewResult with allow/deny decision.
        """
        if not self._enabled:
            return ReviewResult(
                allow=True,
                risk=RiskLevel.LOW,
                summary="FearAgent disabled",
                reviewed_at=datetime.utcnow().isoformat(),
            )

        findings: list[Finding] = []

        # Action-specific checks
        if action_type == "shell_exec":
            cmd = action_params.get("command", "")
            findings.extend(self._check_shell_command(cmd))
        elif action_type == "file_write":
            path = action_params.get("path", "")
            findings.extend(self._check_file_write(path))
        elif action_type == "network_request":
            url = action_params.get("url", "")
            findings.extend(self._check_network_request(url))

        risk = self._calculate_risk(findings)
        allow = self._should_allow(findings, risk)

        return ReviewResult(
            allow=allow,
            risk=risk,
            findings=findings,
            summary=f"Action review: {action_type}",
            reviewed_at=datetime.utcnow().isoformat(),
        )

    def _analyze_with_llm(
        self,
        code: str,
        source: str,
        context: str,
    ) -> list[Finding]:
        """Use LLM to analyze code for security issues."""
        if not self._llm:
            return []

        prompt = self._build_review_prompt(code, source, context)

        try:
            # Use deterministic settings for consistent reviews
            result = self._llm.generate_json(
                prompt,
                temperature=0.0,
                max_tokens=2048,
            )

            if not result:
                return []

            findings = []
            for item in result.get("findings", []):
                try:
                    category_str = item.get("category", "unknown")
                    try:
                        category = DangerCategory(category_str)
                    except ValueError:
                        category = DangerCategory.UNKNOWN

                    severity_str = item.get("severity", "medium")
                    try:
                        severity = RiskLevel(severity_str)
                    except ValueError:
                        severity = RiskLevel.MEDIUM

                    findings.append(
                        Finding(
                            category=category,
                            description=item.get("description", ""),
                            location=item.get("location", source),
                            severity=severity,
                            evidence=item.get("evidence", ""),
                        )
                    )
                except (ValueError, KeyError):
                    continue

            return findings

        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return []

    def _build_review_prompt(self, code: str, source: str, context: str) -> str:
        """Build the prompt for LLM code review."""
        # Truncate code if too long
        max_code_len = 30000
        if len(code) > max_code_len:
            code = code[:max_code_len] + "\n... [truncated]"

        return f"""You are a security reviewer analyzing code for potential risks.
Review the following code and identify any security concerns.

Source: {source}
Context: {context or "Code loaded for execution"}

Code to review:
```
{code}
```

Analyze for these categories:
- code_execution: subprocess, eval, exec, os.system, dynamic imports
- network_access: socket, requests, urllib, any network calls
- file_system: file writes, deletes, permission changes
- data_exfiltration: encoding + network, suspicious data handling
- privilege_escalation: sudo, setuid, capability changes
- persistence: cron, systemd, autostart mechanisms
- obfuscation: base64 decode+exec, marshalled code, pickle loads

Return JSON with this structure:
{{
    "findings": [
        {{
            "category": "category_name",
            "description": "what the code does",
            "location": "line or function name",
            "severity": "low|medium|high|critical",
            "evidence": "relevant code snippet"
        }}
    ],
    "overall_risk": "low|medium|high|critical",
    "summary": "brief summary of findings"
}}

If no issues found, return: {{"findings": [], "overall_risk": "low", "summary": "No security concerns identified"}}
"""

    def _check_shell_command(self, cmd: str) -> list[Finding]:
        """Check a shell command for dangerous patterns."""
        findings = []
        dangerous = [
            ("rm -rf", "Recursive delete"),
            ("chmod 777", "World-writable permissions"),
            (r"curl.*\|.*sh", "Remote script execution"),
            (r"wget.*\|.*sh", "Remote script execution"),
            ("> /dev/sd", "Direct disk write"),
            ("dd if=", "Direct disk operation"),
            ("mkfs", "Filesystem formatting"),
            (":(){:|:&};:", "Fork bomb"),
        ]

        for pattern, desc in dangerous:
            if re.search(pattern, cmd, re.IGNORECASE):
                findings.append(
                    Finding(
                        category=DangerCategory.CODE_EXECUTION,
                        description=desc,
                        evidence=cmd,
                        severity=RiskLevel.HIGH,
                    )
                )

        return findings

    def _check_file_write(self, path: str) -> list[Finding]:
        """Check a file write path for dangerous locations."""
        findings = []
        dangerous_paths = [
            ("/etc/", "System configuration"),
            ("/usr/", "System binaries"),
            ("/bin/", "System binaries"),
            ("/sbin/", "System binaries"),
            (".bashrc", "Shell configuration"),
            (".profile", "Shell configuration"),
            (".ssh/", "SSH configuration"),
            ("cron", "Scheduled tasks"),
            ("systemd", "System services"),
            ("/boot/", "Boot configuration"),
        ]

        for dangerous, desc in dangerous_paths:
            if dangerous in path:
                findings.append(
                    Finding(
                        category=DangerCategory.PERSISTENCE,
                        description=f"Write to sensitive location ({desc}): {dangerous}",
                        location=path,
                        severity=RiskLevel.HIGH,
                    )
                )

        return findings

    def _check_network_request(self, url: str) -> list[Finding]:
        """Check a network request URL."""
        findings = []

        # Check for direct IP addresses
        if re.match(r"https?://\d+\.\d+\.\d+\.\d+", url):
            findings.append(
                Finding(
                    category=DangerCategory.NETWORK_ACCESS,
                    description="Direct IP address connection",
                    evidence=url,
                    severity=RiskLevel.MEDIUM,
                )
            )

        # Check for non-standard ports
        port_match = re.search(r":(\d+)", url)
        if port_match:
            port = int(port_match.group(1))
            if port not in (80, 443, 8080, 8443):
                findings.append(
                    Finding(
                        category=DangerCategory.NETWORK_ACCESS,
                        description=f"Non-standard port: {port}",
                        evidence=url,
                        severity=RiskLevel.LOW,
                    )
                )

        return findings

    def _calculate_risk(self, findings: list[Finding]) -> RiskLevel:
        """Calculate overall risk from findings."""
        if not findings:
            return RiskLevel.LOW

        severities = [f.severity for f in findings]

        if RiskLevel.CRITICAL in severities:
            return RiskLevel.CRITICAL
        if severities.count(RiskLevel.HIGH) >= 2:
            return RiskLevel.CRITICAL
        if RiskLevel.HIGH in severities:
            return RiskLevel.HIGH
        if severities.count(RiskLevel.MEDIUM) >= 3:
            return RiskLevel.HIGH
        if RiskLevel.MEDIUM in severities:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def _should_allow(self, findings: list[Finding], risk: RiskLevel) -> bool:
        """Determine if action should be allowed."""
        if self._strict_mode and findings:
            return False

        # Block critical and high risk by default
        if risk in (RiskLevel.CRITICAL, RiskLevel.HIGH):
            return False

        return True

    def _summarize_findings(self, findings: list[Finding]) -> str:
        """Create a human-readable summary of findings."""
        if not findings:
            return "No security concerns identified."

        by_category: dict[DangerCategory, int] = {}
        for f in findings:
            by_category[f.category] = by_category.get(f.category, 0) + 1

        parts = [f"{cat.value}: {count}" for cat, count in by_category.items()]
        return f"Found {len(findings)} concerns: " + ", ".join(parts)
