#!/bin/sh
# Release Clefkit locally: set the version, run the tests, check staged files for private paths, then commit.
# It does not push; push the commit yourself.
#
# Usage: scripts/release.sh <version> <message file>
#   The message file holds the commit message; its first line should start with the version.
set -eu

[ $# -eq 2 ] || { echo "usage: $0 <version> <message file>" >&2; exit 2; }
version=$1
message=$2
here=$(git rev-parse --show-toplevel)
cd "$here"

sed -i.bak "s/\"version\": \"[^\"]*\"/\"version\": \"$version\"/" plugin/.claude-plugin/plugin.json package.json
rm -f plugin/.claude-plugin/plugin.json.bak package.json.bak
if ! grep -q "\"version\": \"$version\"" plugin/.claude-plugin/plugin.json; then echo "version not set" >&2; exit 1; fi

if ! node --test 'plugin/tests/*.test.ts'; then echo "plugin tests failed" >&2; exit 1; fi
if ! python3 -m unittest discover -s server/tests; then echo "server tests failed" >&2; exit 1; fi

git add -A plugin server worker docs scripts .claude-plugin README.md package.json LICENSE .gitignore
# This machine's home, volume and temp paths, built here so the script never matches itself.
private="/Users/$(id -un)/|/Volumes/|/private/tmp/|claude-$(id -u)"
if git grep --cached -nE "$private" -- . ':!scripts/release.sh'; then
  echo "a private path is staged" >&2
  exit 1
fi
git diff --cached --check
git commit -q -F "$message"
echo "committed $(git log --oneline -1); push it yourself"
