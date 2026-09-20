#!/bin/sh
# docker compose run --rm app <script> [args]   ->   python -m scripts.<script> [args]
set -e
if [ $# -eq 0 ]; then
    echo "usage: <script> [args]; available scripts:"
    ls scripts/*.py | grep -v __init__ | sed "s#scripts/##; s#\.py##; s#^#  #"
    exit 1
fi
name="${1%.py}"; shift
exec python -m "scripts.${name}" "$@"
