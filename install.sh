#!/usr/bin/env bash
set -euo pipefail

prefix=""
while [[ $# -gt 0 ]]; do
	case "$1" in
		--prefix)
			prefix="$2"; shift 2 ;;
		--prefix=*)
			prefix="${1#*=}"; shift ;;
		*)
			echo "Unknown option: $1" >&2
			exit 1
	esac
done

if [[ -z "$prefix" ]]; then
	echo "Usage: $0 --prefix <target-directory>" >&2
	exit 1
fi

src="$(dirname "$0")/cp"
dst="$prefix/include/cp"

rm -rf "$dst"
mkdir -p "$dst"
cp -r "$src/"* "$dst/"

echo "Installed to $dst"
