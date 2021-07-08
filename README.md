# Semantic Segmentation

## Semantic Segmentation Library
This project leverages 2 RTX A6000 GPUs and Tramac's semantic segmentation library to place color coded overlays onto objects given the input of an image view from a car's dashboard.

The model used in this semantic segmentation application is a PSPNet model that is trained using a backbone of Resnet50 for 500 epochs. 

## Image Processing Script
The semantic-segmentation.py script is an image processing Python script that allows the user to input an image, a folder containing images or a video and feeds these into the trained neural network model to get a color coded overlay output. 

The user can use the --img (input an image path), --vid (input a video path) and --flder (input a folder path) flags to specify the type of input you want the program to use. You can also use the -r flag with the --vid flag in order to specify the frame rate for the converted image frames. 

After the application gets the resulting segmented images, the program writes these images to a folder within the semantic segmentation repository for the user to view. 