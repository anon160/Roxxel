# Agent Instructions

## Versioning

Version is derived from git tags via `hatch-vcs`. No manual version in `pyproject.toml`.

- `git tag vX.Y.Z && git push origin main --tags` sets the release version.
- Between tags, builds get an auto `.devN+hash` suffix.

## Workflow

When modifying code, always:

1. Run `python -m pytest tests/ -x -q` before committing.
2. Update `AGENTS.md` with any new conventions or recurring steps introduced.
