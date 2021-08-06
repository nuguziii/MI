# Deep Learning in Medical Imaging

This is a baseline code for reproducible deep learning research and fast model development for medical image segmentation task. It provides implementations for 3d image-tools (i.e., augmentation, edge extractor), models (i.e., 3DUNet, 3DDenseUNet), losses (i.e., cross-entropy, Dice loss), evaluation and utils (e.g., nifti I/O). It also provides an example of Liver-Tumor segmentation task.

## Getting Started

- [tasks/LiverTumorSegmentation](https://github.com/nuguziii/MI/tree/develop/tasks/LiverTumorSegmentation) Is train & test code for Liver-Tumor segmentation task using [LiTS](https://competitions.codalab.org/competitions/17094) dataset
- [src/image_tools](https://github.com/nuguziii/MI/tree/develop/src/image_tools) Is image tools for 2D/3D augmentation, image processing (i.e., edge detection) and transformation
- [src/loss](https://github.com/nuguziii/MI/tree/develop/src/loss) Is implementation of loss functions (i.e., Cross-entropy, Dice loss and L1)
- [src/network](https://github.com/nuguziii/MI/tree/develop/src/network) Is implementation of models (i.e., 3DUNet, DenseUNet, DenseUNet3D)

### Training:

Before training, you need to set tasks/LiverTumorSegmentation/config.yaml

```
python tasks/LiverTumorSegmentation/main.py
```

## Training on Your Own Dataset

To train the model on your own dataset, you'll need to create a new directory in `tasks/` and follow examples in `tasks/LiverTumorSegmentation`

`config.yaml` contains the default configuration. Modify the attributes you need to change.

`main.py` code for train/test model. Copy&Paste this code in your task.
