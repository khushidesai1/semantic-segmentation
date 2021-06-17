# Semantic Segmentation

## Image Reader
Python program that reads images in the given folder. The program looks for one argument -- the file path to the folder containing the images to be read. 

The program errors if it finds an invalid image format within the folder and if the folder path is not valid.

Run the program using the following format:
```
python3 image_reader.py [path to folder]
```
An example to read images from the folder ~/images_source:
```
python3 image_reader.py ~/images_source
```