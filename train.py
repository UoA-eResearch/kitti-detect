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

def main():

    # experiment name corresponds to the foldername and file prefix for data partitions and label maps
    exp_name = "t1-final"
    batch_size = 16
    epochs = 100
    use_label_overrides = True
    train_path = f"{exp_name}/{exp_name}-part-0.json"
    test_path = f"{exp_name}/{exp_name}-test.json"

    # load data
    with open(train_path, 'r') as f:
        data_train = json.load(f)

    with open(test_path, 'r') as f:
        data_test = json.load(f)

    print(f"train samples: {len(data_train)}, test samples: {len(data_test)}")

    # load the label map
    label_map_path = f"{exp_name}/{exp_name}-label-map.json"
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    print(f"label map: {label_map}")
    reverse_label_map = {v: k for k, v in label_map.items()}

    # define datasets and data loaders
    batch_size = 16
    ds = KITTIDataset(data_train, label_map, use_label_overrides=use_label_overrides, transforms=get_transforms(train=True))
    ds_test = KITTIDataset(data_test, label_map, use_label_overrides=use_label_overrides, transforms=get_transforms(train=False))

    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=utils.collate_fn)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=1, collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')
    print('device:', device)

    model = get_model(len(label_map)).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001)

    t_start = time.time()
    logs = []
    for e in range(epochs):
        logger = train_one_epoch(model, optimizer, dl, device, e, print_freq=100, grad_clip=1)
        log_values = {k:{'median':v.median, 'mean':v.avg} for k,v in logger.meters.items()}
        logs.append(log_values)
        # update the learning rate
        #lr_scheduler.step()
        # evaluate on the test dataset
        #evaluate(model, dl_test, device=device)
    print(f'total time: {(time.time() - t_start)/3600} hrs')

    # save logs
    log_name = f"{exp_name}/{exp_name}-model-logs.json"
    with open(log_name, 'w') as f:
        json.dump(logs, f)    

    # save the model
    model_path = f"{exp_name}/{exp_name}-model.pt"
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main()