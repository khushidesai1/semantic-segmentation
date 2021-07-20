# Semantic Segmentation Project

This project has 2 main sections:
* Semantic Segmentation Script
* Image Differential Generator

## Semantic Segmentation Script

### Segmentation Library and Hardware
This project leverages 2 RTX A6000 GPUs and Tramac's [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch) library to place color coded overlays onto objects given the input of an image view from a car's dashboard.

The model used in this semantic segmentation application is a PSPNet model that is trained using a backbone of Resnet50 for 2500 epochs. 

### Segmentation Script Overview
The semantic-segmentation.py script is an image processing Python script that allows the user to input an image, a folder containing images or a video and feeds these into the trained neural network model to get a color coded overlay output. 

The user can use the --img (input an image path), --vid (input a video path) and --flder (input a folder path) flags to specify the type of input you want the program to use. You can also use the -r flag with the --vid flag in order to specify the frame rate for the converted image frames. 

After the application gets the resulting segmented images, the program writes these images to a folder within the semantic segmentation repository for the user to view. 

### Usage and Examples

**Input a single image**
Follow the below format to input a single image file and replace the directory to the image with your own
file path.
```
python semantic-segmentation.py --img [directory to image file]

# Example -- replace ./test-image.jpg with your own image path
python semantic-segmentation.py --img ./test-image.jpg
```

**Input a video**
Follow the below format to input a video file and replace the directory to the video with your own file path.
The frame rate at which the video will be converted to image frames can also be specified (for example, 0.03 is approximately 33 frames per second). If the frame rate is not specified, the program will use a default frame rate of 0.05 (20 fps).
```
python semantic-segmentation.py --vid [directory to video file] -r [frame rate in decimals]

# Example -- replace ./test-vid.mp4 with your own video path and 0.02 with your own preferred frame rate
python semantic-segmentation.py --vid ./test-vid.mp4 -r 0.02
```

**Input a folder of images**
Follow the below format to input a folder containing images and replace the directory to the folder with your
own folder path.
```
python semantic-segmentation.py --flder [directory to folder]

# Example -- replace ./test-folder with your own folder path
python semantic-segmentation --flder ./test-folder
```

### Output Format
You can find all outputs to the program within the ./runs folder. Each run is given a random string code and the results of the segmentation is stored in these folders.

**Image output**
The image outputs will be stored as: ./runs/[string code]/[image name]-seg.[image extension]

**Video output**
The video outputs will be stored as: ./runs/[string code]/[video name]-seg.[video extension]

**Folder output**
The folder containing image outputs will be stored within the following folder: ./runs/[string code]

## Image Differential Generator

