#!/bin/bash

mkdir -p combined/resized/val/images
mkdir -p combined/resized/val/labels

for exid in 602658 571134 15427c 16060d 571657 600032 574065 36940b 8625b 600973 600066 37420a; do
    mv "combined/resized/images/${exid}.jpg" combined/resized/val/images
    mv "combined/resized/labels/${exid}.txt" combined/resized/val/labels
done

