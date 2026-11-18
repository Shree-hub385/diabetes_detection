#!/usr/bin/env bash
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
if [ -z "$PY" ]; then echo "Python 3.9+ not found. Install from https://www.python.org/downloads/"; exit 1; fi
exec "$PY" run.py "$@"
