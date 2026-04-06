#!/bin/sh

rm -rf env

if python3 -m venv env; then
    if [ -f "requirements.txt" ]; then
        ./env/bin/python3 -m pip install -r requirements.txt
    else
        echo "requirements.txt not found."
    fi
    # Activate the environment
    . env/bin/activate
else
    echo "Error: Failed to create the virtual environment."
    return 1 2>/dev/null || exit 1
fi
