#!/bin/bash
# Convenience script to run training
# Usage: ./train.sh dataset=qm9 [other options]

# Ensure we are in the project root
cd "$(dirname "$0")"

# Set PYTHONPATH to current directory so 'src' module can be found
export PYTHONPATH=.

# Run training using the virtual environment python
# Pass all arguments to the python script
.venv/bin/python src/main.py "$@"
