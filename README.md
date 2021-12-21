# SRGAN
A PyTorch implementation of SRGAN based on CVPR 2017 paper 
[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision -c pytorch
```
- opencv
```
conda install opencv
```


### Recreate .pth file
```
python train.py

optional arguments:
--crop_size                   training images crop size [default value is 88, value used in assignment is 78]
--upscale_factor              super resolution upscale factor, value used in assignment is 3)
--num_epochs                  train epoch number [default value is 100, epoch used in assignment 1400]
```
The output val super resolution images are on `training_results` directory.
Remember to put training images in the appropriate directory, and edit directory path in train.py.

### Recreate results file
```
python test_image.py

optional arguments:
--upscale_factor              super resolution upscale factor [default value is 4]
--test_mode                   using GPU or CPU [default value is 'GPU'](choices:['GPU', 'CPU'])
--image_name                  test low resolution image name
--model_name                  generator model epoch name [default value is netG_epoch_4_100.pth]
```
The output super resolution image is in HW4_results folder, if it does not exist, mkdir it. 

### Test Single Video
```
python test_video.py

optional arguments:
--upscale_factor              super resolution upscale factor [default value is 4]
--video_name                  test low resolution video name
--model_name                  generator model epoch name [default value is netG_epoch_4_100.pth]
```
The output super resolution video and compared video are on the same directory.


