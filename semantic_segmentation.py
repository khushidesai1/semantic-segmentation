from PIL import Image
import sys
import os
import copy
import argparse
import cv2
					
def read_images(img_path=None, folder_path=None, vid_path=None, frame_rate=0.5):
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
	frame_rate: double
		The desired frame rate to conver the video to frame images
	
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
		convert_vid(vid_path, frame_rate)

def convert_vid(vid_path, frame_rate):
	"""
	Converts a video into image frames and saves it to the current directory in the folder 
	./[video file path]-frames/

	Parameters
	----------
	vid_path: str
		The path to the video file to be converted
	frame_rate: double
		The desired frame rate when capturing pictures from the video
	
	"""
	vidcap = cv2.VideoCapture(vid_path)
	sec = 0
	success = get_frame(vidcap, sec, vid_path)
	while success:
		sec = round(sec + frame_rate, 2)
		success = get_frame(vidcap, sec, vid_path)

def get_frame(vidcap, sec, vid_path):
	"""
	Gets a single frame using the VideoCapture object and saves the frame image to the
	folder ./[video file path]-frames/

	Parameters
	----------
	vidcap: VideoCapture object
		The VideoCapture object read from the video file path
	sec: double
		The second value of the video to capture
	vid_path: str
		The video file path

	Returns
	-------
	boolean
		True if the function was able to save the frame image to the folder
		and False otherwise

	"""
	vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
	hasFrames, img = vidcap.read()
	if hasFrames:
		cv2.imwrite("./frames/image" + str(sec) + ".jpg", img)
	return hasFrames

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
	usage="semantic_segmentation.py [ --img | --vid | --fldr ] [ image path I | video path V | folder path F ]")
	flag_group = parser.add_mutually_exclusive_group(required=True)
	flag_group.add_argument("--img", help="Use this flag when passing in an image file path I")
	flag_group.add_argument("--vid", help="Use this flag when passing in a video file path V")
	parser.add_argument("-r", help="Use this flag to specify the frame rate to convert the video to frame images")
	flag_group.add_argument("--fldr", help="Use this flag when passing in a folder path containing images F")
	return parser

def main():
	"""
	Processes the arguments, reads the given image, images in the folder or the given video,
	passes the images into the trained semantic segmentation model, and writes the output to a folder.
	"""
	parser = build_parser()
	input_args = parser.parse_args()
	image_path = input_args.img
	video_path = input_args.vid
	folder_path = input_args.fldr
	rate = float(input_args.r)
	data = read_images(img_path=image_path, folder_path=folder_path, vid_path=video_path, frame_rate=rate)

if __name__ == "__main__":
	main()