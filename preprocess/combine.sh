#!/bin/bash

rm -rf "${1}/combined"
mkdir -p "${1}/combined/images"
mkdir -p "${1}/combined/labels"

find "${1}" -iname "*.jpg" -exec cp {} "${1}/combined/images" \;
find "${1}" -iname "*.txt" -exec cp {} "${1}/combined/labels" \;
