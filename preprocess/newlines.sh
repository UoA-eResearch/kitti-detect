#!/bin/bash

find ${1}/combined/resized/labels/ -name "*.txt" -exec sed -i -e '$a\' {} \;
