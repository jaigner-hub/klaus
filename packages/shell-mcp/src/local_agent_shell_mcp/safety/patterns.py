"""Shell safety classification: allow / confirm / deny."""

import re
import shlex
from typing import Literal

Classification = Literal["allow", "confirm", "deny"]

# Matched against the FULL command string before any other check.
# Any match → unconditional deny, regardless of operator chaining context.
_DENY_PATTERNS = [
    # rm with recursive flag on root or critical system directories
    r"\brm\b.*-[a-zA-Z]*r[a-zA-Z]*[fF][a-zA-Z]*\s+(/\s*$|/\s|/(etc|boot|sys|proc|dev|usr|bin|sbin|lib)\b)",
    r"\brm\b.*-[a-zA-Z]*[fF][a-zA-Z]*r[a-zA-Z]*\s+(/\s*$|/\s|/(etc|boot|sys|proc|dev|usr|bin|sbin|lib)\b)",
    r"\brm\b.*--recursive\b.*\s(/\s*$|/(etc|boot|sys|proc|dev|usr|bin|sbin|lib)\b)",
    # dd writing to block devices (not /dev/null)
    r"\bdd\b.*\bof=/dev/(?!null\b)",
    # output redirect to /dev/ (not null) or system config dirs
    r">+\s*/dev/(?!null\b)",
    r">+\s*/(etc|boot|sys|proc)\b",
    # find with flags that delete files
    r"\bfind\b.*\s--?delete\b",
    r"\bfind\b.*\s-exec\s+rm\b",
    # filesystem and disk tools
    r"\bmkfs\b",
    r"\bshred\b",
    r"\bfdisk\b",
    r"\bparted\b",
    # fork bomb
    r":\s*\(\s*\)\s*\{",
    # download directly to devices
    r"\b(wget|curl)\b.*\s-[oO]\s+/dev/(?!null\b)",
]

_DENY_COMPILED = [re.compile(p) for p in _DENY_PATTERNS]

# Any of these in the command → require human confirmation.
# Covers pipes, chaining operators, redirects, and subshell execution.
# The denylist is checked first, so patterns there take precedence.
_SHELL_META_RE = re.compile(r"&&|\|\||;|\||>|`|\$\(")

# Commands that are always safe: read-only, no side effects.
_SAFE_BASE_COMMANDS = frozenset({
    "ls", "cat", "head", "tail",
    "grep", "egrep", "fgrep",
    "rg",
    "find",
    "wc", "echo", "pwd", "which", "type",
    "sort", "uniq",
    "diff",
    "file", "stat", "du", "df",
    "date", "uname", "hostname",
    "id", "whoami",
    "env", "printenv",
})

# git subcommands that are safe (read-only inspection only).
_SAFE_GIT_SUBCOMMANDS = frozenset({
    "status", "log", "diff", "show",
    "branch", "remote", "tag", "describe",
    "shortlog", "blame", "annotate",
    "ls-files", "ls-remote",
    "stash",
})


def classify(command: str) -> Classification:
    """Classify a shell command as 'allow', 'confirm', or 'deny'.

    deny   — matches a catastrophic denylist pattern; rejected unconditionally.
    confirm — contains shell metacharacters or an unknown base command;
              surface a y/n prompt to the user before executing.
    allow  — a known-safe read-only command with no shell metacharacters.
    """
    stripped = command.strip()

    # 1. Hard denylist: scan the full command string (catches injection after &&, ;, etc.)
    for pattern in _DENY_COMPILED:
        if pattern.search(stripped):
            return "deny"

    # 2. Shell metacharacters: require confirmation even if individual commands look safe
    if _SHELL_META_RE.search(stripped):
        return "confirm"

    # 3. Parse to find the base command
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        return "confirm"

    if not tokens:
        return "confirm"

    base = tokens[0]

    # 4. git: allow only read-only inspection subcommands
    if base == "git":
        subcommand = next(
            (t for t in tokens[1:] if not t.startswith("-")),
            "",
        )
        return "allow" if subcommand in _SAFE_GIT_SUBCOMMANDS else "confirm"

    # 5. Known-safe base command
    if base in _SAFE_BASE_COMMANDS:
        return "allow"

    return "confirm"
