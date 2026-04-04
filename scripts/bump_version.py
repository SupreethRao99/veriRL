#!/usr/bin/env python3
"""Auto-detect semantic version bump from changelog fragments."""
import os
import sys
from pathlib import Path

def get_current_version(pyproject_path):
    """Extract version from pyproject.toml."""
    with open(pyproject_path) as f:
        for line in f:
            if line.startswith('version = "'):
                return line.split('"')[1]
    raise ValueError("Could not find version in pyproject.toml")

def get_fragment_types(changelog_dir):
    """Get all fragment types in .changelog.d/."""
    has_feature = False
    has_bugfix = False

    for frag in Path(changelog_dir).glob("*.md"):
        if frag.name in ("README.md", "template.md"):
            continue
        if ".feature." in frag.name:
            has_feature = True
        elif ".bugfix." in frag.name:
            has_bugfix = True

    return has_feature, has_bugfix

def bump_version(current_version, has_feature, has_bugfix):
    """Calculate next semantic version."""
    parts = current_version.split(".")
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

    if has_feature:
        # Feature = minor bump (x.Y.z)
        minor += 1
        patch = 0
    elif has_bugfix:
        # Bugfix = patch bump (x.y.Z)
        patch += 1
    else:
        # No changes, just patch bump
        patch += 1

    return f"{major}.{minor}.{patch}"

if __name__ == "__main__":
    root = Path(__file__).parent.parent
    pyproject = root / "pyproject.toml"
    changelog_dir = root / ".changelog.d"

    current = get_current_version(pyproject)
    has_feature, has_bugfix = get_fragment_types(changelog_dir)

    next_version = bump_version(current, has_feature, has_bugfix)
    print(next_version)
