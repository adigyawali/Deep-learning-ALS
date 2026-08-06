#!/usr/bin/env bash
#
# ONE-TIME MIGRATION — strip the `_run-NN` / `_runN` tag from already-processed
# sample folders.
#
# Preprocessing used to keep the rerun tag in the processed name
# (`Data/processed/CALSNIC2_EDM_C034_V2_run-02/`). It no longer does: a rerun is
# the same subject at the same timepoint, so it now writes
# `Data/processed/CALSNIC2_EDM_C034_V2/`. Without this migration the next
# preprocessing run creates the untagged folder alongside the tagged one and the
# dataset loader reads BOTH as two separate samples of the same scan.
#
# This renames rather than reprocesses, because the volumes themselves are
# unchanged — only their names were wrong. That saves re-running ANTs on every
# affected subject.
#
# Usage:
#   scripts/migrate_processed_run_tags.sh                 # dry run (default)
#   DRY=0 scripts/migrate_processed_run_tags.sh           # actually rename
#   DRY=0 scripts/migrate_processed_run_tags.sh /path/to/processed
#
# Read the dry-run output before running with DRY=0. A folder whose untagged
# name already exists is skipped and reported, never overwritten.

set -euo pipefail

DRY="${DRY:-1}"
TARGET="${1:-Data/processed}"

run() {
    if [ "$DRY" = "1" ]; then
        echo "  would: mv $1 -> $2"
    else
        mv "$1" "$2"
    fi
}

if [ ! -d "$TARGET" ]; then
    echo "ERROR: $TARGET is not a directory." >&2
    exit 1
fi

cd "$TARGET"
shopt -s nullglob

if [ "$DRY" = "1" ]; then
    echo "DRY RUN (nothing is changed). Re-run with DRY=0 to apply."
fi

n=0
for d in *_run-*/ *_run[0-9]*/; do
    d="${d%/}"
    new="$(printf '%s' "$d" | sed -E 's/_run-?[0-9]+//')"
    if [ -e "$new" ]; then
        echo "  SKIP (target already exists): $d -> $new"
        continue
    fi
    for f in "$d"/*; do
        run "$f" "$d/$(basename "$f" | sed -E 's/_run-?[0-9]+//')"
    done
    run "$d" "$new"
    n=$((n + 1))
done

for f in _QC_Snapshots/*_run-* _QC_Snapshots/*_run[0-9]*; do
    new="$(printf '%s' "$f" | sed -E 's/_run-?[0-9]+//')"
    if [ -e "$new" ]; then
        echo "  SKIP (target already exists): $f -> $new"
        continue
    fi
    run "$f" "$new"
done

echo "Done. $n sample folder(s) $([ "$DRY" = "1" ] && echo 'would be' || echo 'were') renamed."
echo "manifest.csv is rebuilt automatically by the next preprocessing run."
