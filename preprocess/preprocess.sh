#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: preprocess.sh path/to/data"
    exit 1
fi

bash combine.sh "${1}"
bash relabel.sh "${1}"
bash rename.sh "${1}"
bash rotate.sh "${1}"
bash resize.sh "${1}"
bash newlines.sh "${1}"
bash classes.sh "${1}"



