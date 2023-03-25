#!/bin/bash

cd phase_vocoder
venv_name="env"
venv_path="./$venv_name"

if [ ! -d "$venv_path" ]; then
    virtualenv "$venv_path"
fi

source "$venv_path/bin/activate"

if ! pip freeze | grep -q -F -x -f requirements.txt; then
    pip install -r requirements.txt
fi

python phase_vocoder.py "$@"