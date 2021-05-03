#!/bin/bash

find "${1}"/combined -depth -name "*.JPG" -exec sh -c 'mv "$1" "${1%.JPG}.jpg"' _ {} \;

