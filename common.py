from PIL import Image, ExifTags
import os
import torch
import torch.utils.data
import json

class KITTIDataset(torch.utils.data.Dataset):

    def __init__(self, records, label_map, use_label_overrides=False, transforms=None):
        super().__init__()
        self.transforms = transforms
        self.records = records
        self.label_map = label_map
        self.use_label_overrides = use_label_overrides

        if (self.use_label_overrides):
            print("Warning, label overrides assume one object per image")

    def get_image_path(self, idx):
        return self.records[idx]['image_path']

    def get_label_path(self, idx):
        return self.records[idx]['label_path']

    def get_label_override(self, idx):
        return self.records[idx]['label_override']

    def __getitem__(self, idx):

        try:
            # load images and bounding boxes
            image_path = self.get_image_path(idx)
            label_path = self.get_label_path(idx)
            label_override = self.get_label_override(idx)

            img = Image.open(image_path).convert("RGB")
            objects = parse_kitti(label_path)
            boxes = torch.tensor([o['bounds'] for o in objects], dtype=torch.float32)

            # relabel if needed
            if self.use_label_overrides:
                labels = torch.tensor([self.label_map[label_override] for o in objects], dtype=torch.int64)
            else:
                labels = torch.tensor([self.label_map[o['label']] for o in objects], dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = torch.tensor([idx])
            target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target["iscrowd"] = torch.zeros((len(objects),), dtype=torch.int64)

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target

        except Exception as e:
            print(self.records[idx])
            raise e

    def __len__(self):
        return len(self.records)

def parse_kitti(path, label_overrides=None):
    # https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md
    objects = []
    with open(path, 'r') as f:
        name, ext = os.path.splitext(os.path.basename(path))
        for line in [l.strip() for l in f.readlines()]:
            label, _, _, _, xmin, ymin, xmax, ymax, *_ = line.split()
            if label_overrides and name in label_overrides:
                label = label_overrides[name]
            objects.append({'label': label, 'bounds': [int(float(x)) for x in [xmin, ymin, xmax, ymax]]})
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