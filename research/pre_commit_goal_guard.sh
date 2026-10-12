#!/usr/bin/env bash
# Git pre-commit hook: block commits that change the research contract unless the
# user approves with GOAL_CHANGE_APPROVED=1. Install once per machine:
#   ln -sf ../../research/pre_commit_goal_guard.sh .git/hooks/pre-commit
set -euo pipefail
protected=(GOAL.md research/frozen_artifacts.sha256 research/pre_commit_goal_guard.sh)
changed="$(git diff --cached --name-only -- "${protected[@]}")"
if [[ -n "$changed" && "${GOAL_CHANGE_APPROVED:-}" != "1" ]]; then
    echo "Blocked: these files need the user's approval to change:" >&2
    echo "$changed" | sed 's/^/  /' >&2
    echo "Propose the change in WORKLOG.md instead. The user commits it with GOAL_CHANGE_APPROVED=1." >&2
    exit 1
fi
