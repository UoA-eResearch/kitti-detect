#!/bin/bash

classes=$(cat "${1}"/combined/resized/labels/*.txt | cut -d " " -f1 | sort | uniq | tr '\n' ',')
echo ${classes::-1}
