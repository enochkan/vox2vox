# vox2vox
Pytorch implementation of vox2vox, a 3D volume-to-volume generative adversarial network proposed by [M. Cirillo, D. Abramian and A. Eklund](https://arxiv.org/abs/2003.13653)

### Prerequisites
* Windows/ Linux/ Mac
* torch>=0.4.0
* torchvision
* matplotlib
* numpy, scipy
* PIL
* scikit-image
* pyvista (for visulization)
* h5py

### Data preprocessing

All volumes should be saved as h5 files since the dataloader uses h5py to load and create training batches. 

Datasets should be split into two folders `train` and `test`, in which they should be placed under the same folder in `/data`. Each of the `train` and `test` folder should have image and segmentation volumes with the same file name. Suggested extension for image volume is `.im` and `.seg` for segmentation volume.

### Getting started

Clone this repo by:
`git clone https://github.com/chinokenochkan/vox2vox`

Run training job:
`cd vox2vox && python train.py --dataset <your dataset name>`

### Parameters

Here are some of the possible parameters you can modify during training:

* `--n_epochs`: number of epochs to train for
* `--glr`: generator learning rate for Adam optimizer
* `--dlr`: discriminator learning rate for Adam optimizer
* `--dataset_name`: name of the dataset you are using, should be consistent with the name of the dataset folder in `/data`
* `--batch_size`: training batch size, default is 1
* `--image_height`/ `--image_width`/ `--image_depth`: image dimensions h/w/d
* `d_threshold`: accuracy threshold in which the discriminator will be trained as long as its accuracy is below it

### Acknowledgement
Thanks @eriklindernoren for the great starter [repo](https://github.com/eriklindernoren/PyTorch-GAN)

Oringal paper: [M. Cirillo, D. Abramian and A. Eklund](https://arxiv.org/abs/2003.13653)