#!/bin/bash

width=300
height=300
cvdata_resize --input_images "${1}/combined/images" --input_annotations "${1}/combined/labels" --output_images "${1}/combined/resized/images" --output_annotations "${1}/combined/resized/labels" --width $width --height $height --format kitti

