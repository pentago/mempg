#!/usr/bin/env bash
# pre-commit pre-push hook: rejects tag pushes where the tag does not equal the
# package.json version ON THE COMMIT THE TAG POINTS TO. Mirrors the publish
# workflow's version-match requirement, but fails at `git push` time instead of
# wasting a GitHub Actions run (and an npm release cycle) on the mismatch.
#
# pre-commit forwards git's pre-push stdin: "<local_ref> <local_oid> ..." per ref.
set -uo pipefail

status=0
while read -r local_ref local_oid _; do
  case "$local_ref" in
    refs/tags/v*) ;;
    *) continue ;;
  esac

  tag="${local_ref#refs/tags/}"
  want="${tag#v}"
  declared=$(
    git show "$local_oid":package.json |
      sed -n 's/^[[:space:]]*"version":[[:space:]]*"\([^"]*\)".*/\1/p'
  )

  if [ -z "$declared" ]; then
    echo "tag-version-match: could not read package.json version from $tag ($local_oid)" >&2
    status=1
  elif [ "$declared" != "$want" ]; then
    echo "tag-version-match: refusing push of $tag - package.json on that commit declares $declared." \
      "Bump the version on the tagged commit, then re-tag." >&2
    status=1
  fi
done

exit "$status"
