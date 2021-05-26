import pandas as pd
from glob import glob
import random
import os
import json

def parse_kitti(path, label_overrides=None):
    # https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md
    objects = []
    with open(path, 'r') as f:
        name, ext = os.path.splitext(os.path.basename(path))
        for line in [l.strip() for l in f.readlines()]:
            label, _, _, _, xmin, ymin, xmax, ymax, *_ = line.split()
            if label_overrides and name in label_overrides:
                label = label_overrides[name]
            objects.append({'label': label, 'bounds': [int(x) for x in [xmin, ymin, xmax, ymax]]})
    return objects

def class_summary(paths, label_dir, label_overrides):
    classes = {}
    for im in paths:
        image_id, _ = os.path.splitext(os.path.basename(im))
        label_path = os.path.join(label_dir, f"{image_id}.txt")
        objects = parse_kitti(label_path, label_overrides=label_overrides)
        for o in objects:
            x1, y1, x2, y2 = o['bounds']
            if x2 - x1 == 0 or y2 - y1 == 0:
                print(f"Error: {im} non-positive BBOX dimensions: {[x1, y1, x2, y2]}")
            else:
                label = o['label']
                if label in classes:
                    classes[label].append((im, label_path))
                else:
                    classes[label] = [(im, label_path)]
    return classes

def distribution(classes):
    distribution = {k:len(v) for k,v in classes.items()}
    total = sum([len(v) for k,v in classes.items()])
    print('total:', total)
    for c, n in distribution.items():
        print(f"{c}: {n}, {n/total:.2f}")

def save_partition(name, classes):
    records = []
    for label, samples in classes.items():
        for image_path, label_path in samples:
            records.append({'label_override':label, 'image_path': image_path, 'label_path': label_path})
    with open(name, 'w') as f:
        json.dump(records, f)