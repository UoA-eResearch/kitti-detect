# kitti-detect
Object detection using KITTI format

## Installation

Anaconda is recommended, create a Python environment e.g.
```
conda create -n kitti python=3.7
```

Once you have an appropriate environment, activate it and install the dependencies:

```
conda activate kitti
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -c conda-forge pycocotools
conda install numpy matplotlib pillow jupyter scikit-image
```

Note pycocotools is not required for the kitti_check.ipynb workflow.

Then start a Jupyter Notebook session and run kitti_detect.ipynb

## Data Generation

Use https://github.com/UoA-eResearch/annotate-to-KITTI or similar to generate KITTI training data with images and labels in separate folders.
