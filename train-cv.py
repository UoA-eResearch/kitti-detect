import time
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T

from common import KITTIDataset
import json
from glob import glob
import numpy as np

def get_transforms(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
    return T.Compose(transforms)


def get_model(num_classes, pretrained=False):
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def load_partitions(pattern):
    parts = []
    paths = glob(pattern)
    for p in paths:
        with open(p, 'r') as f:
            parts.append(json.load(f))
    return parts

def train_test_partitions(parts, test_idx):
    p_test = parts[test_idx]
    p_train = []
    for i in range(len(parts)):
        if i != test_idx:
            p_train += parts[i]
    return p_train, p_test

def iou(boxA, boxB):
    # from: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def eval_model(model, ds, label_map, device):
    """
    Get top-1 classification and IoU metrics
    """
    reverse_label_map = {v: k for k, v in label_map.items()}
    results = np.zeros((len(ds), 2))

    model.eval()
    with torch.no_grad():
        for idx in range(len(ds)):
            image, target = ds[idx]
            pred = model([image.to(device)])
            if len(pred) > 0 and len(pred[0]['boxes']) > 0:
                # sort predicitons by classifacation score
                labels, scores, boxes = pred[0]['labels'], pred[0]['scores'], pred[0]['boxes']
                pred = list(zip(labels, scores, boxes))
                pred.sort(key=lambda x: x[1], reverse=True)

                true_label = reverse_label_map[target['labels'][0].item()]
                true_box = target['boxes'][0].int().tolist()

                pred_labels = [reverse_label_map[l.item()] for l, s, _ in pred]
                pred_boxes = [b for _, _, b in pred]

                pred_box = pred_boxes[0].int().tolist()
                iou_score = iou(true_box, pred_box)

                correct = pred_labels[0] == true_label
                results[idx] = np.array([correct, iou_score])

    return results.mean(axis=0)

def main():
    # experiment name corresponds to the foldername and file prefix for data partitions and label maps
    exp_name = "t1-final"
    batch_size = 16
    lr = 0.001
    epochs = 100
    print_freq = 100
    use_label_overrides = True

    # load partitions
    pattern = f"{exp_name}/{exp_name}-part-*.json"
    partitions = load_partitions(pattern)
    print(f"folds: {len(partitions)}, samples: {sum([len(p) for p in partitions])}")

    # load the label map
    label_map_path = f"{exp_name}/{exp_name}-label-map.json"
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    print(f"label map: {label_map}")
    reverse_label_map = {v: k for k, v in label_map.items()}

    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    print('device:', device)

    # cross validation
    for i in range(len(partitions)):
        test_fold = i  # index of the test partition
        data_train, data_test = train_test_partitions(partitions, test_fold)
        print(f"test_fold: {test_fold}, train samples: {len(data_train)}, test samples: {len(data_test)}")

        # define datasets and data loaders
        ds = KITTIDataset(data_train, label_map, use_label_overrides=use_label_overrides,
                          transforms=get_transforms(train=True))
        ds_test = KITTIDataset(data_test, label_map, use_label_overrides=use_label_overrides,
                               transforms=get_transforms(train=False))
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=1,
                                         collate_fn=utils.collate_fn)
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=1,
                                              collate_fn=utils.collate_fn)

        # create model and optimizer
        model = get_model(len(label_map)).to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)

        # train model
        t_start = time.time()
        logs = []
        for e in range(epochs):
            logger = train_one_epoch(model, optimizer, dl, device, e, print_freq=100, grad_clip=1)
            # evaluate(model, dl_test, device=device)
            train_acc, train_iou = eval_model(model, ds, label_map, device)
            test_acc, test_iou = eval_model(model, ds_test, label_map, device)
            print(f"acc: {train_acc:.2f} ({test_acc:.2f}), iou: {train_iou:.2f}, ({test_iou:.2f})")
            log_values = {k: {'mean': v.avg} for k, v in logger.meters.items()}
            log_values['train_acc'] = {'mean': train_acc}
            log_values['train_iou'] = {'mean': train_iou}
            log_values['test_acc'] = {'mean': test_acc}
            log_values['test_iou'] = {'mean': test_iou}
            logs.append(log_values)
        print(f'time: {(time.time() - t_start) / 3600:.2f} hrs')

        # save logs
        log_name = f"{exp_name}/{exp_name}-model-logs-{test_fold}.json"
        with open(log_name, 'w') as f:
            json.dump(logs, f)

        print("========================================================")

if __name__ == "__main__":
    main()
