# Semantic Segmentation Project

This project leverages 2 RTX A6000 GPUs and Tramac's [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch) library to place color coded overlays onto objects given the input of an image view from a car's dashboard.

This project has 3 main sections:
* Semantic Segmentation Training
* Semantic Segmentation Evaluation Script
* Image Differential Generator

### Set Up Instructions
**Dataset Set Up**: This program uses the Cityscapes dataset and the dataset needs to be set up within the system in order for the scripts to use it. You can install the leftImg8bit_trainvaltest.zip (11GB) dataset from the [Cityscapes website](https://www.cityscapes-dataset.com/login/) along with it's ground truth dataset, gtFine_trainvaltest.zip (241MB). In order to ensure that the folders are correctly placed, follow the directory structure below:
```
./awesome-semantic-segmentation-pytorch
|-- datasets
|   |-- citys
|   |   |-- leftImg8bit
|   |   |   |-- train
|   |   |   |-- val
|   |   |   |-- test
|   |   |-- gtFine
```

## Semantic Segmentation Training
**Set Up Training**: In order to set up the training component, run the following command.
```
python setup.py
```
### Semantic Segmentation Model Training

In order to perform semantic segmentation, the program looks for a pre-trained model within the ~/.torch/models folder
in the system the user is using. This folder is only created once a model is trained.

### Usage and Examples

Follow the below general format to perform training using a chosen model, backbone and the Cityscapes dataset.
```
python train.py --model [ model ] --backbone [ backbone ] --lr [ loss rate ] --epochs [ epochs ] --ngpus [ number of GPUS to use ]

Example using PSPNet model and ResNet50 backbone
python train.py --model psp --backbone resnet50 --lr 0.0001 --epochs 500 --ngpus 4
```

## Semantic Segmentation Evaluation Script

### Segmentation Script Overview

The default model used in this semantic segmentation application is a PSPNet model that is trained using a backbone of Resnet50.

See the **Usage and Examples** section to process input dash cam images using a custom model and backbone.

The semantic-segmentation.py script is an image processing Python script that allows the user to input an image, a folder containing images or a video and feeds these into the trained neural network model to get a color coded overlay output. 

The user can use the --img (input an image path), --vid (input a video path) and --flder (input a folder path) flags to specify the type of input you want the program to use. You can also use the -r flag with the --vid flag in order to specify the frame rate for the converted image frames. You can use the --mask flag with the --img flag in order to obtain metrics on an input image by passing in an additional mask image. (*Note: the input mask has to be one from the Cityscapes dataset*)

After the application gets the resulting segmented images, the program writes these images to a folder within the semantic segmentation repository for the user to view. 

### Usage and Examples

**Input a single image**:
Follow the below format to input a single image file and replace the path to the image with your own
file path.
```
python semantic_segmentation.py --img [path to image file]

# Example -- replace ./test-image.jpg with your own image path
python semantic_segmentation.py --img ./test-image.jpg
```

**Input a single image and obtain metrics**:
In order to obtain metrics on the evaluation result, pass in a Cityscapes mask .png image corresponding to a Cityscapes input image using the --mask flag.
```
python semantic_segmentation.py --img [path to Cityscapes image] --mask [path to Cityscapes mask]

# Example -- replace ./test-image.png and ./test-mask.png with your own Cityscapes image and mask
python semantic_segmentation.py --img ./test-image.png --mask ./test-mask.png
```

**Input a video**:
Follow the below format to input a video file and replace the path to the video with your own file path.
The frame rate at which the video will be converted to image frames can also be specified (for example, 0.03 is approximately 33 frames per second). If the frame rate is not specified, the program will use a default frame rate of 0.05 (20 fps).
```
python semantic_segmentation.py --vid [path to video file] -r [frame rate in decimals]

# Example -- replace ./test-vid.mp4 with your own video path and 0.02 with your own preferred frame rate
python semantic_segmentation.py --vid ./test-vid.mp4 -r 0.02
```

**Input a folder of images**:
Follow the below format to input a folder containing images and replace the path to the folder with your
own folder path.
```
python semantic_segmentation.py --dir [path to directory]

# Example -- replace ./test-folder with your own folder path
python semantic_segmentation.py --dir ./test-folder
```

**Evaluating using multi-GPU**:
You can use multi-GPU evaluation by using the above commands and adding a flag to input the number of GPUs.
```
python semantic_segmentation.py --dir ./test-folder --ngpus [ number of GPUS ]

Example using 4 GPUs for evaluation
python semantic_segmentation.py --dir ./test-folder --ngpus 4
```

**Evaluating on custom model and backbone**:
You can perform semantic segmentation using a custom model and backbone based on your pre-trained model.
```
python semantic_segmentation.py --dir ./test-folder --model [ model name ] --backbone [ backbone name ]

Example using FCN23s model and ResNet101 backbone
python semantic_segmentation.py --dir ./test-folder --model fcn32s --backbone resnet101
```

### Output Format
You can find all outputs to the program within the ./runs folder. Each run is given a random string code and the results of the segmentation is stored in these folders.

**Image output**:
The image outputs will be stored as: semantic-segmentation/runs/[string code]/[image name]-seg.[image extension]

**Video output**:
The video outputs will be stored as: semantic-segmentation/runs/[string code]/[video name]-seg.[video extension]

**Folder output**:
The folder containing image outputs will be stored within the following folder: ./runs/[string code]

## Image Differential Generator

### Image Differential Overview

The image_diff.py script is an image differential generator. Given the input of 2 file paths corresponding to images, the program will calculate the difference between the images and output the SSIM, the percentage of difference and will save the differential and threshold images within a diffs folder.

### Usage and Examples

Use the --first and the --second flags to input the first and second image for comparison. Replace the paths to the images with the paths to your own images as shown in the example below.
```
python image_diff.py --first [path to first image] --second [path to second image]

# Example
python image_diff.py --first ./test-image1.jpg --second ./test-image2.jpg
```

### Output Format
You can find all outputs to the image differential program in the ./diffs folder. Each set of outputs is  stored in a folder named [first image name]-[second image name]. The two output images will be stored as:
* ./diffs/[first image name]-[second image name]/image-diff.png
* ./diffs/[first image name]-[second image name]/image-thresh.png