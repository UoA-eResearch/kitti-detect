# kitti-detect
Object detection using KITTI format

## Installation

Anaconda is recommended, create a Python environment e.g.
```
conda create -n kitti python=3.7
```

Dependencies depend on whether you need to do full object detection or simply verify the input data

### Object Detection Workflow

Once you have an appropriate environment, activate it and install the dependencies:

```
conda activate kitti
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -c conda-forge pycocotools
conda install numpy matplotlib pillow jupyter scikit-image
```

Then start a Jupyter Notebook session and run kitti_detect.ipynb

### Data Checking Workflow
If you only need to check the KITTI formatted data samples, you can install fewer dependencies:
```
conda activate kitti
conda install numpy matplotlib pillow jupyter
```
Then start a Jupyter Notebook session and run kitti_check.ipynb

## Data Generation

Use https://github.com/UoA-eResearch/annotate-to-KITTI or similar to generate KITTI training data with images and labels in separate folders.
