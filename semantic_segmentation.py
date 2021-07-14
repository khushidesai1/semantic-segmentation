from PIL import Image
import sys
import os
import argparse
import cv2
import random
import string

def apply_segmentation_dir(image_paths, dest_path):
	"""
	Takes in image paths from a directory and feeds each image into the trained Tramac neural 
	network model. Writes the resulting overlays for each image to a specified directory.
	
	Parameters
	----------
	image_paths: list of strs
		The list of paths to the images within a certain directory
	dest_path: str
		The path to the destination directory where the segmented images will be stored
	"""
	pass

def apply_single_segmentation(img_path, dest_path):
	"""
	Takes in a single image path and feeds the image into the trained Tramac neural network model.
	Writes the resulting overlay image to the specified directory path. 

	Parameters
	----------
	img_path: str
		The path to the given image
	dest_path: str
		The path to the destination directory where the segmented image will be stored
	"""
	pass
					
def process_images(img_path=None, dir_path=None, vid_path=None, frame_rate=0.5):
	"""
	Reads the given image, images in the given directory, or image frames from the given video.
	Based on the format of the input, processes the image(s) by applying segmentation to each
	video and saving the results in a directory.

	The results will be stored in the following formats:
	- For a single image/directory of images: ./runs/[random string code]/[image name]-seg.jpg
	- For a video: ./runs/[random string code]/[video name]-seg.mp4

	Parameters
	----------
	img_path: str, optional
		The path to the given image
	dir_path: str, optional
		The path to the given directory
	vid_path: str, optional
		The path to the given video
	frame_rate: double
		The desired frame rate to conver the video to frame images
	"""
	run_id = generate_id()
	dest_path = "./runs/" + run_id + "/"
	if img_path: 
		validate_img(img_path)
		print("Completed reading image:", img_path)
		dest_path += get_file_name(img_path) + "-seg." + get_file_extension(img_path)
		apply_single_segmentation(img_path, dest_path)
	if dir_path:
		img_paths = validate_dir(dir_path)
		for img_name in img_paths:
			validate_img(dir_path + "/" + img_name)
		print("Completed reading images from directory:", dir_path)
		apply_segmentation_dir(img_paths, dest_path)
	if vid_path:
		frames_dir = './' + get_file_name(vid_path) + '-frames'
		vid_to_frames(vid_path, frames_dir, frame_rate)
		frames_per_second = 1 / frame_rate
		print("Completed converting video to frames at", frames_per_second, "frames per second")
		frame_paths = validate_dir(frames_dir)
		dest_path += get_file_name(vid_path) + "-seg." + get_file_extension(vid_path)
		apply_segmentation_dir(frame_paths, dest_path)
	
def frames_to_vid(frames_dir_path, frame_rate):
	"""
	Converts provided frame images to a video and saves it to the 
	"""

def vid_to_frames(vid_path, frames_dir, frame_rate):
	"""
	Converts a video into image frames and saves it to the given frames directory.

	Parameters
	----------
	vid_path: str
		The path to the video file to be converted
	frames_dir: str
		The path to the directory where the frames from the video will be stores
	frame_rate: double
		The desired frame rate when capturing pictures from the video
	
	Returns
	-------
	str
		The directory path containing the image frames converted from the video

	"""
	if not os.path.isdir(frames_dir):
		os.mkdir(frames_dir)
	vidcap = cv2.VideoCapture(vid_path)
	sec = 0
	success = get_frame(vidcap, sec, frames_dir)
	while success:
		sec = round(sec + frame_rate, 2)
		success = get_frame(vidcap, sec, frames_dir)
	return frames_dir

def get_frame(vidcap, sec, frames_dir):
	"""
	Gets a single frame using the VideoCapture object and saves the frame image to the
	directory ./[video name]-frames/

	Parameters
	----------
	vidcap: VideoCapture object
		The VideoCapture object read from the video file path
	sec: double
		The second value of the video to capture
	frames_dir: str
		The path to the directory in which the video frames will be stored

	Returns
	-------
	boolean
		True if the function was able to save the frame image to the directory
		and False otherwise

	"""
	vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
	hasFrames, img = vidcap.read()
	if hasFrames:
		cv2.imwrite(frames_dir + "/image" + str(sec) + ".jpg", img)
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

def validate_dir(dir_path):
	"""
	Checks whether the given directory path corresponds to a valid directory or not.

	Parameters
	----------
	dir_path: str
		The path to the directory

	Returns
	-------
	list of str
		The list of strs representing each image file from the images within the directory path
	
	"""
	try:
		img_paths = os.listdir(dir_path)
		return img_paths
	except:
		print("Invalid Argument:", dir_path, "is not a valid directory")

def generate_id():
	"""
	Generates a random 9 letter alphanumeric string and returns it.

	Returns
	-------
	str
		The random generated string
	"""
	return ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=8))

def get_file_name(file_path):
	"""
	Takes in a file path and returns the name of the file without the extension and the directories
	the path is contained in.

	Parameters
	----------
	file_path: str
		The given file path
	
	Returns
	-------
	str
		The name of the inputted file path
	"""
	return file_path.split('/')[-1].split('.')[0]

def get_file_extension(file_path):
	"""
	Takes in a file path and returns the extension of the file.

	Parameters
	----------
	file_path: str
		The given file path
	
	Returns
	-------
	str
		The extension of the inputted file path
	"""
	return file_path.split('.')[-1]

def parse_args():
	"""
	Builds a parser with 3 flags to accept a single image, a dir of images and a video file.
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
		usage="semantic_segmentation.py [ --img | --vid | --dir ] [ image path I | video path V | directory path D ]")
	flag_group = parser.add_mutually_exclusive_group(required=True)
	flag_group.add_argument("--img", type='str', help="Use this flag when passing in an image file path I")
	flag_group.add_argument("--vid", type='str', help="Use this flag when passing in a video file path V")
	parser.add_argument("-r", type='float', help="Use this flag to specify the frame rate to convert the video to frame images")
	flag_group.add_argument("--dir", type='str', help="Use this flag when passing in a dir path containing images D")
	return parser.parse_args()

def main():
	"""
	Processes the arguments, reads the given image, images in the directory or the given video,
	passes the images into the trained semantic segmentation model, and writes the output to a directory.
	"""
	input_args = parse_args()
	image_path = input_args.img
	video_path = input_args.vid
	dir_path = input_args.dir
	rate = None
	if input_args.r:
		rate = float(input_args.r)
	data = process_images(img_path=image_path, dir_path=dir_path, vid_path=video_path, frame_rate=rate)

if __name__ == "__main__":
	main()