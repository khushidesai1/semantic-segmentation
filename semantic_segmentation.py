from PIL import Image
import sys
import os
import copy
import argparse
					
def read_images(img_path=None, folder_path=None, vid_path=None):
	"""
	Reads the given image, images in the given folder, or image frames from the given video
	and outputs them in the form of an image object or a list of image objects.

	Parameters
	----------
	img_path: str, optional
		The path to the given image
	folder_path: str, optional
		The path to the given folder
	vid_path: str, optional
		The path to the given video
	
	Returns
	-------
	list of Image objects or Image object
	"""
	if img_path:
		img = validate_img(img_path)
		print("Completed reading image:", img_path)
		return img
	if folder_path:
		img_paths = validate_folder(folder_path)
		img_list = []
		for img_name in img_paths:
			img_list.append(copy.deepcopy(validate_img(folder_path + "/" + img_name)))
		print("Completed reading images from folder:", folder_path)
		return img_list
	if vid_path:
		# process video into frames
		pass

def validate_img(file_path):
	"""
	Checks whether the given file path corresponds to a valid image or not.
	
	Parameters
	----------
	file_path: str
		The path to the image file
		
	Returns
	-------
	Image object
		The Image object from the file path
	"""
	try:
		img = Image.open(file_path)
		return img
	except:
		print("Value Error:", file_path, "is not a valid image file")
		sys.exit(1)

def validate_folder(folder_path):
	"""
	Checks whether the given folder path corresponds to a valid folder or not.

	Parameters
	----------
	folder_path: str
		The path to the directory

	Returns
	-------
	list of str
		The list of strs representing each image file from the images within the folder path
	"""
	try:
		img_paths = os.listdir(folder_path)
		return img_paths
	except:
		print("Invalid Argument: the folder", folder_path, "is not a valid directory")

def validate_video(video_path):
	"""
	Checks whether the given video path corresponds to a valid video or not.

	Parameters
	----------
	video_path: str
		The path to the video file

	Returns
	-------
	Video object
		The video file as a Video object
	"""
	pass

def build_parser():
	"""
	Builds a parser with 3 flags to accept a single image, a folder of images and a video file.
	Parser ensures that at least one of the arguments is provided and that not more than one of them
	is provided in the terminal. Parser includes a description of how to use the flags and a help
	description. 

	Returns
	-------
	ArgumentParser
		The parser object containing all the specified configurations
	"""
	parser = argparse.ArgumentParser(
	description="Process images using trained semantic segmentation model.",
	usage="semantic_segmentation.py [ -i | -v | -f ] [ image path I | video path V | folder path F ]")
	flag_group = parser.add_mutually_exclusive_group(required=True)
	flag_group.add_argument("-i", help="Use this flag when passing in an image file path I")
	flag_group.add_argument("-v", help="Use this flag when passing in a video file path V")
	flag_group.add_argument("-f", help="Use this flag when passing in a folder path containing images F")
	return parser

def main():
	"""
	Processes the arguments, reads the given image, images in the folder or the given video,
	passes the images into the trained semantic segmentation model, and writes the output to a folder.
	"""
	parser = build_parser()
	input_args = parser.parse_args()
	image_path = input_args.i
	video_path = input_args.v
	folder_path = input_args.f
	data = read_images(img_path=image_path, folder_path=folder_path, vid_path=video_path)

if __name__ == "__main__":
	main()