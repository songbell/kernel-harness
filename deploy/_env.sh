#!/usr/bin/env bash
# _env.sh -- read a key from the gitignored .env. Sourced by the deploy scripts so the
# parsing rules are stated once.
#
# Value = everything after the first '='. A trailing inline comment (KEY=value  # note) and
# surrounding quotes are tolerated, so a .env copied verbatim from .env.example works. A
# value that needs a literal '#' or a trailing space must be quoted. A shell export of the
# same name wins over the file.

env_get() {
  local key="$1"
  # Environment beats file, so `CKH_REMOTE_HOST=box bash deploy/...` works as expected.
  if [[ -n "${!key:-}" ]]; then printf '%s' "${!key}"; return 0; fi
  [[ -r "$REPO_ROOT/.env" ]] || return 0
  local line
  line="$(grep -E "^[[:space:]]*$key=" "$REPO_ROOT/.env" 2>/dev/null | head -1)" || return 0
  line="${line#*=}"
  line="${line#"${line%%[![:space:]]*}"}"                 # ltrim
  if [[ "$line" == \"* ]]; then
    line="${line#\"}"; line="${line%%\"*}"
  elif [[ "$line" == \'* ]]; then
    line="${line#\'}"; line="${line%%\'*}"
  else
    if [[ "$line" == \#* ]]; then line=""; else line="${line%%[[:space:]]#*}"; fi
    line="${line%"${line##*[![:space:]]}"}"               # rtrim
  fi
  printf '%s' "$line"
}
