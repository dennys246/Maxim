"""Coding guidelines for LLM-generated code.

Provides language-specific best practices that are injected into prompts
when users request scripts or files in various languages/technologies.
"""

from __future__ import annotations

import re
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Language Detection Keywords
# ─────────────────────────────────────────────────────────────────────────────

# Keywords that indicate specific language/technology requests
LANGUAGE_KEYWORDS: dict[str, list[str]] = {
    "python": [
        "python", "py", ".py", "script", "pip", "virtualenv", "venv",
        "pandas", "numpy", "flask", "django", "fastapi", "pytest",
    ],
    "html": [
        "html", "webpage", "web page", "website", "markup", ".html",
        "css", "stylesheet", "bootstrap", "tailwind",
    ],
    "javascript": [
        "javascript", "js", ".js", "node", "nodejs", "npm", "react",
        "vue", "angular", "typescript", "ts", ".ts", "express",
    ],
    "docker": [
        "docker", "dockerfile", "container", "containerize", "docker-compose",
        "compose", "image", "kubernetes", "k8s",
    ],
    "bash": [
        "bash", "shell", "sh", ".sh", "terminal", "command line", "cli",
        "zsh", "scripting",
    ],
    "yaml": [
        "yaml", "yml", ".yaml", ".yml", "config", "configuration",
        "ansible", "helm", "github actions", "ci/cd", "pipeline",
    ],
    "json": [
        "json", ".json", "api response", "data format",
    ],
    "sql": [
        "sql", "database", "query", "mysql", "postgres", "sqlite",
        "select", "insert", "update", "table",
    ],
    "markdown": [
        "markdown", "md", ".md", "readme", "documentation", "docs",
        "research document", "research paper", "document", "report",
    ],
    "glob": [
        "glob", "find file", "find files", "search file", "search files",
        "pattern", "wildcard", "**", "*.py", "*.js", "*.html",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Language-Specific Guidelines
# ─────────────────────────────────────────────────────────────────────────────

CODING_GUIDELINES: dict[str, str] = {
    "python": """=== PYTHON BEST PRACTICES ===
Structure:
- Use descriptive variable/function names (snake_case)
- Add a docstring explaining what the script does
- Put main logic in a `if __name__ == "__main__":` block
- Import standard library first, then third-party, then local

Error Handling:
- Use try/except for operations that might fail (file I/O, network, parsing)
- Provide helpful error messages

Code Quality:
- Keep functions small and focused (one responsibility)
- Use type hints for function parameters and returns
- Avoid global variables; pass data through function parameters

Common Patterns:
```python
#!/usr/bin/env python3
\"\"\"Brief description of what this script does.\"\"\"

def main():
    # Main logic here
    pass

if __name__ == "__main__":
    main()
```""",

    "html": """=== HTML/CSS BEST PRACTICES ===
Structure:
- Always include <!DOCTYPE html> declaration
- Use semantic HTML5 elements (<header>, <nav>, <main>, <footer>, <article>, <section>)
- Include <meta charset="UTF-8"> and viewport meta tag for responsiveness

Accessibility:
- Add alt text to images
- Use appropriate heading hierarchy (h1 -> h2 -> h3)
- Ensure sufficient color contrast
- Add aria labels where needed

CSS:
- Use CSS variables for colors and repeated values
- Mobile-first responsive design with media queries
- Prefer flexbox or grid for layouts
- Avoid inline styles; use classes

Template:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Page Title</title>
    <style>
        :root { --primary-color: #007bff; }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: system-ui, sans-serif; line-height: 1.6; }
    </style>
</head>
<body>
    <main>
        <!-- Content here -->
    </main>
</body>
</html>
```""",

    "javascript": """=== JAVASCRIPT BEST PRACTICES ===
Modern JS:
- Use const by default, let when reassignment needed (never var)
- Use arrow functions for callbacks: (x) => x * 2
- Use template literals: `Hello ${name}`
- Use destructuring: const { name, age } = person

Error Handling:
- Use try/catch for async operations
- Always handle Promise rejections (.catch() or try/catch with async/await)

Code Quality:
- Use meaningful variable names (camelCase)
- Avoid global variables; use modules
- Prefer async/await over raw Promises for readability

Node.js Specific:
- Use ES modules (import/export) or CommonJS (require) consistently
- Handle process signals (SIGINT, SIGTERM) for graceful shutdown
- Use environment variables for configuration

Pattern:
```javascript
// ES Module style
async function main() {
    try {
        // Main logic
    } catch (error) {
        console.error('Error:', error.message);
        process.exit(1);
    }
}

main();
```""",

    "docker": """=== DOCKER BEST PRACTICES ===
Dockerfile:
- Use official base images (python:3.11-slim, node:20-alpine)
- Use specific version tags, not :latest
- Order commands from least to most frequently changed (for layer caching)
- Combine RUN commands to reduce layers
- Use .dockerignore to exclude unnecessary files
- Run as non-root user for security
- Use multi-stage builds for smaller images

Template:
```dockerfile
# Stage 1: Build
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app
# Create non-root user
RUN useradd --create-home appuser
USER appuser
# Copy from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
CMD ["python", "main.py"]
```

Docker Compose:
- Use version: '3.8' or later
- Define healthchecks
- Use named volumes for persistent data
- Set restart policies""",

    "bash": """=== BASH SCRIPTING BEST PRACTICES ===
Header & Safety:
- Start with shebang: #!/usr/bin/env bash
- Use 'set -euo pipefail' for safer scripts:
  - -e: Exit on error
  - -u: Error on undefined variables
  - -o pipefail: Catch pipe failures

Variables:
- Quote variables: "$var" not $var
- Use ${var:-default} for defaults
- Use readonly for constants

Functions:
- Use local variables inside functions
- Return exit codes (0 for success)

Template:
```bash
#!/usr/bin/env bash
set -euo pipefail

# Description: What this script does
# Usage: ./script.sh [args]

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

main() {
    local arg="${1:-default}"
    echo "Running with: $arg"
}

main "$@"
```

Common Patterns:
- Check if command exists: command -v cmd &> /dev/null
- Check if file exists: [[ -f "$file" ]]
- Loop over files: for f in *.txt; do ... done""",

    "yaml": """=== YAML BEST PRACTICES ===
Formatting:
- Use 2-space indentation (consistent throughout)
- Use lowercase keys with hyphens or underscores
- Quote strings containing special characters
- Use anchors (&) and aliases (*) for repeated values

Structure:
- Group related configuration together
- Add comments for non-obvious settings
- Use environment variable substitution where supported

GitHub Actions Example:
```yaml
name: CI Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest
```

Docker Compose Example:
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=${DATABASE_URL}
    depends_on:
      - db
    restart: unless-stopped
```""",

    "json": """=== JSON BEST PRACTICES ===
Formatting:
- Use 2-space indentation for readability
- Keep keys in consistent order (alphabetical or logical grouping)
- Use camelCase or snake_case consistently for keys

Structure:
- Use meaningful key names
- Nest logically related data
- Use arrays for lists of similar items
- Use null for missing optional values

API Response Pattern:
```json
{
  "success": true,
  "data": {
    "id": 123,
    "name": "Example",
    "items": []
  },
  "meta": {
    "timestamp": "2025-01-01T00:00:00Z",
    "version": "1.0"
  }
}
```

Note: JSON does not support comments. Use a separate schema file or documentation.""",

    "sql": """=== SQL BEST PRACTICES ===
Formatting:
- Use UPPERCASE for SQL keywords (SELECT, FROM, WHERE)
- Use lowercase for table/column names
- Use meaningful table and column names
- Indent for readability

Security:
- ALWAYS use parameterized queries to prevent SQL injection
- Never concatenate user input into SQL strings
- Use least privilege for database users

Query Structure:
```sql
SELECT
    u.id,
    u.name,
    COUNT(o.id) AS order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.active = true
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 0
ORDER BY order_count DESC
LIMIT 10;
```

Python Example (parameterized):
```python
cursor.execute(
    "SELECT * FROM users WHERE id = ?",
    (user_id,)  # Parameters as tuple
)
```""",

    "markdown": """=== MARKDOWN & RESEARCH DOCUMENT GUIDELINES ===
Structure:
- Start with a clear title (# Title)
- Use heading hierarchy (# > ## > ### )
- Add a brief description/abstract after the title
- Include table of contents for long documents

Formatting:
- Use blank lines between sections
- Use code blocks with language specifiers (```python)
- Use bullet points for lists, numbers for sequences
- Keep lines under 100 characters for readability

=== RESEARCH DOCUMENT STRUCTURE ===
When writing research documents, reports, or informational content:

1. Opening Paragraph: Provide a concise introduction that defines the topic and
   establishes its significance. This should give the reader context and frame
   what follows.

2. Feature Elaboration (2-3 Key Points): Select 2-3 of the most important or
   distinctive aspects of the topic. For each feature, write a dedicated
   paragraph that:
   - Names and clearly defines the feature
   - Explains HOW it works or functions in practice
   - Describes WHY it matters (benefits, implications, use cases)
   - Provides a concrete example or application if relevant

3. Closing: Summarize the key takeaways or provide forward-looking perspective.

4. Data & References: When possible, incorporate supporting evidence:
   - Include relevant statistics, metrics, or quantitative data
   - Use APA-style citations for sources (Author, Year)
   - Reference graphs, charts, or visualizations when they would clarify concepts
   - Link to primary sources or further reading when available

Target Length: ~900 characters for focused research responses. Longer documents
should expand feature paragraphs proportionally.

Research Response Template:
```markdown
# [Topic Title]

[Opening paragraph: Define topic, establish context and significance]

## Key Features

### [Feature 1 Name]
[Paragraph explaining what it is, how it works, and why it matters]

### [Feature 2 Name]
[Paragraph with definition, mechanism, and practical implications]

### [Feature 3 Name] (optional)
[Paragraph covering the third most important aspect]

## Summary
[Brief concluding statement with key takeaways]
```""",

    "glob": """=== GLOB PATTERN GUIDE ===
Glob patterns are used to match files and directories. Use the 'glob' tool to find files.

**SANDBOX**: Search in '.maxim_sandbox/' for sandbox files:
- pattern='.maxim_sandbox/**/*.py' (all Python files in sandbox)

Pattern Syntax:
- *        → matches any characters EXCEPT /
- **       → matches any characters INCLUDING / (recursive)
- ?        → matches exactly one character
- [abc]    → matches a, b, or c
- [a-z]    → matches any lowercase letter
- {a,b}    → matches a OR b
- !pattern → excludes matches (in some contexts)

Common Patterns:
- Find all Python files:        '**/*.py'
- Find in specific folder:      'src/**/*.ts'
- Multiple extensions:          '**/*.{js,ts,jsx,tsx}'
- Config files:                 '**/{config,settings}*.{json,yaml,yml}'
- Test files:                   '**/test_*.py' or '**/*_test.py'
- Hidden files:                 '**/.*' (use include_hidden=True)
- Specific depth:               '*/*.py' (only one level deep)
- All files in folder:          'src/*' (files only) or 'src/**/*' (recursive)

Sandbox Examples:
- '.maxim_sandbox/*.py'         → Python files directly in sandbox
- '.maxim_sandbox/**/*.py'      → All Python files recursively
- '.maxim_sandbox/scripts/*'    → Everything in scripts folder

Tips:
- Start broad, then narrow down if too many results
- Use max_results param to limit output
- Combine with read_file to examine matched files""",
}

# ─────────────────────────────────────────────────────────────────────────────
# Detection and Injection Functions
# ─────────────────────────────────────────────────────────────────────────────


def detect_languages(text: str) -> list[str]:
    """Detect which programming languages/technologies are mentioned in text.

    Args:
        text: User input text to analyze

    Returns:
        List of detected language keys (e.g., ["python", "docker"])
    """
    if not text:
        return []

    text_lower = text.lower()
    detected = []

    for lang, keywords in LANGUAGE_KEYWORDS.items():
        for keyword in keywords:
            # Use word boundary matching for short keywords
            if len(keyword) <= 3:
                pattern = rf'\b{re.escape(keyword)}\b'
                if re.search(pattern, text_lower):
                    detected.append(lang)
                    break
            elif keyword in text_lower:
                detected.append(lang)
                break

    return detected


def get_coding_guidelines(languages: list[str], max_guidelines: int = 2) -> str:
    """Get coding guidelines for specified languages.

    Args:
        languages: List of language keys
        max_guidelines: Maximum number of guideline sections to include

    Returns:
        Formatted guidelines string
    """
    if not languages:
        return ""

    # Prioritize: take first N detected languages
    selected = languages[:max_guidelines]

    parts = []
    for lang in selected:
        if lang in CODING_GUIDELINES:
            parts.append(CODING_GUIDELINES[lang])

    if not parts:
        return ""

    return "\n\n".join(parts)


def get_guidelines_for_request(user_request: str, max_guidelines: int = 2) -> str:
    """Detect languages in a request and return appropriate guidelines.

    Args:
        user_request: The user's request text
        max_guidelines: Maximum guideline sections to include

    Returns:
        Formatted guidelines string, or empty if no languages detected
    """
    # Check if this is actually a coding/file creation request
    coding_indicators = [
        "write", "create", "make", "generate", "script", "code", "program",
        "file", "build", "develop", "implement",
    ]

    request_lower = user_request.lower()
    is_coding_request = any(indicator in request_lower for indicator in coding_indicators)

    if not is_coding_request:
        return ""

    languages = detect_languages(user_request)
    return get_coding_guidelines(languages, max_guidelines)


def get_file_extension_guidelines(filename: str) -> str:
    """Get guidelines based on a filename's extension.

    Args:
        filename: The filename (e.g., "script.py", "Dockerfile")

    Returns:
        Guidelines string for the file type
    """
    if not filename:
        return ""

    filename_lower = filename.lower()

    # Map extensions to languages
    extension_map = {
        ".py": "python",
        ".html": "html",
        ".htm": "html",
        ".css": "html",  # CSS guidelines are in HTML
        ".js": "javascript",
        ".ts": "javascript",
        ".jsx": "javascript",
        ".tsx": "javascript",
        ".sh": "bash",
        ".bash": "bash",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".sql": "sql",
        ".md": "markdown",
        ".markdown": "markdown",
    }

    # Special filenames
    if "dockerfile" in filename_lower:
        return CODING_GUIDELINES.get("docker", "")
    if "docker-compose" in filename_lower or "compose" in filename_lower:
        return CODING_GUIDELINES.get("docker", "")

    # Check extension
    for ext, lang in extension_map.items():
        if filename_lower.endswith(ext):
            return CODING_GUIDELINES.get(lang, "")

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Sandbox Path Reminder (always include for file operations)
# ─────────────────────────────────────────────────────────────────────────────

SANDBOX_REMINDER = """=== FILE SANDBOX REMINDER ===
All files MUST be written to '.maxim_sandbox/' directory:
- CORRECT: '.maxim_sandbox/script.py', '.maxim_sandbox/index.html'
- WRONG: 'script.py', '/tmp/file.py' (will FAIL!)
"""


def build_coding_context(
    user_request: str,
    include_sandbox_reminder: bool = True,
    max_guidelines: int = 2,
) -> str:
    """Build complete coding context for LLM prompts.

    Args:
        user_request: The user's request
        include_sandbox_reminder: Whether to include sandbox path reminder
        max_guidelines: Maximum guideline sections to include

    Returns:
        Complete coding context string
    """
    parts = []

    # Always include sandbox reminder for file operations
    if include_sandbox_reminder:
        parts.append(SANDBOX_REMINDER)

    # Add language-specific guidelines
    guidelines = get_guidelines_for_request(user_request, max_guidelines)
    if guidelines:
        parts.append(guidelines)

    # Add glob guidelines if searching for files
    if is_file_search_request(user_request):
        glob_guide = get_glob_guidelines()
        if glob_guide and glob_guide not in parts:
            parts.append(glob_guide)

    return "\n\n".join(parts)


def is_file_search_request(text: str) -> bool:
    """Check if the request involves file searching/globbing.

    Args:
        text: User request text

    Returns:
        True if this appears to be a file search request
    """
    if not text:
        return False

    text_lower = text.lower()
    search_indicators = [
        "find file", "find files", "search file", "search files",
        "look for file", "locate file", "list file", "list files",
        "glob", "pattern", "wildcard", "**", "*.py", "*.js",
        "where is", "where are", "which file",
    ]
    return any(indicator in text_lower for indicator in search_indicators)


def get_glob_guidelines() -> str:
    """Get glob pattern guidelines.

    Returns:
        Glob pattern guidelines string
    """
    return CODING_GUIDELINES.get("glob", "")