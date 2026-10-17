#!/bin/sh
# Pristine-base capture and subsequent verification need no Python. Hold only
# native SDK packages; unrelated Python/compiler tools may be added afterward.
set -eu
mode=$1
output=$2
pattern=$3
case "$mode" in
capture)
    mkdir -p "$output"
    dpkg-query -W -f='${binary:Package}\t${Version}\n' > "$output/base-sdk.tsv"
    selected_pattern=$(cat "$pattern")
    awk -v pattern="$selected_pattern" '$1 ~ pattern { print $1 }' "$output/base-sdk.tsv" > "$output/base-sdk-packages.txt"
    test -s "$output/base-sdk-packages.txt"
    xargs -r apt-mark hold < "$output/base-sdk-packages.txt"
    # shellcheck disable=SC2016 # dpkg expands its own format fields.
    xargs -r dpkg-query -W -f='${binary:Package}\t${Version}\n' < "$output/base-sdk-packages.txt" > "$output/base-sdk-versions.raw"
    sort "$output/base-sdk-versions.raw" > "$output/base-sdk-versions.tsv"
    xargs -r dpkg-query -L < "$output/base-sdk-packages.txt" > "$output/base-sdk-paths.raw"
    sort -u "$output/base-sdk-paths.raw" > "$output/base-sdk-paths.txt"
    : > "$output/base-sdk.sha256"
    : > "$output/base-sdk-links.tsv"
    while IFS= read -r file; do
        if test -f "$file"; then sha256sum "$file"; fi
    done < "$output/base-sdk-paths.txt" > "$output/base-sdk.sha256"
    test -s "$output/base-sdk.sha256"
    cp "$output/base-sdk-paths.txt" "$output/base-sdk-aliases.raw"
    printf '%s\n' /usr/local/cuda /etc/alternatives/cuda /etc/alternatives/cuda-13 >> "$output/base-sdk-aliases.raw"
    sort -u "$output/base-sdk-aliases.raw" > "$output/base-sdk-aliases.txt"
    while IFS= read -r file; do
        if test -L "$file"; then
            target=$(readlink -- "$file")
            printf '%s\t%s\n' "$file" "$target"
        fi
    done < "$output/base-sdk-aliases.txt" > "$output/base-sdk-links.tsv"
    ;;
check)
    # shellcheck disable=SC2016 # dpkg expands its own format fields.
    xargs -r dpkg-query -W -f='${binary:Package}\t${Version}\n' < "$output/base-sdk-packages.txt" > "$output/current-sdk-versions.raw"
    sort "$output/current-sdk-versions.raw" > "$output/current-sdk-versions.tsv"
    cmp "$output/base-sdk-versions.tsv" "$output/current-sdk-versions.tsv"
    sha256sum --check --status "$output/base-sdk.sha256"
    while IFS="$(printf '\t')" read -r file target; do
        test -L "$file"
        actual=$(readlink -- "$file")
        test "$actual" = "$target"
    done < "$output/base-sdk-links.tsv"
    ;;
*) echo 'usage: foundation_sdk.sh capture|check OUTPUT PATTERN' >&2; exit 2 ;;
esac
