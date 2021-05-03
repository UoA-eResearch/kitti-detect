#!/bin/bash

for f in "${1}"/combined/labels/*.txt; do
    sed -i 's/.*/\L&/g' $f
done
