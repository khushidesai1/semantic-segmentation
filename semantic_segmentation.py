from posixpath import join
from PIL import Image
import sys
import os
import argparse
import cv2
import random
import string

MODEL = ' --model psp'
BACKBONE = ' --backbone resnet50'
DATASET = ' --dataset citys'

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
	os.chdir('./awesome-semantic-segmentation-pytorch/scripts')
	dest_path = join('../../', dest_path)
	for path in image_paths:
		if not os.path.isdir(dest_path):
			dest_path = os.path.dirname(dest_path)
		
		dest_path = join(dest_path, get_file_name(path) + "-seg" + get_file_extension(path))
		outdir = ' --outdir ' + dest_path
		img = ' --input-pic ' + path
		os.system('python eval_custom.py ' + MODEL + BACKBONE + DATASET + img + outdir)
	os.chdir('../..')

def apply_single_segmentation(img_path, dest_path, mask_path=None):
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
	os.chdir('./awesome-semantic-segmentation-pytorch/scripts')
	dest_path = join('../../', dest_path)
	img = ' --input-pic ' + img_path
	outdir = ' --outdir ' + dest_path
	if not mask_path:
		os.system('python eval_custom.py' + MODEL + BACKBONE + DATASET + img + outdir)
	else:
		mask = ' --input-gt ' + mask_path
		os.system('python eval_custom_metric.py' + MODEL + BACKBONE + DATASET + img + outdir + mask)
	os.chdir('../..')
					
def process_input(img_path=None, dir_path=None, vid_path=None, frame_rate=0.5, mask_path=None):
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
	dest_path = join("./runs", run_id)
	if not os.path.isdir(dest_path):
		os.makedirs(dest_path)
	if img_path: 
		assert os.path.isfile(img_path)
		print("Completed reading image:", img_path)
		dest_path = join(dest_path, get_file_name(img_path) + "-seg" + get_file_extension(img_path))
		apply_single_segmentation(img_path, dest_path, mask_path)
		print("Completed segmentation evaluation. Result is saved as", dest_path)
	if dir_path:
		assert os.path.isdir(dir_path)
		img_paths = [join(dir_path, im) for im in os.listdir(dir_path)]
		for path in img_paths:
			assert os.path.isfile(path)
		print("Completed reading images from directory:", dir_path)
		apply_segmentation_dir(img_paths, dest_path)
		print("Completed segmentation evaluation. Result is saved in", dest_path)
	if vid_path:
		frames_dir = join('./', get_file_name(vid_path) + '-frames')
		vid_to_frames(vid_path, frames_dir, frame_rate)
		frames_per_second = 1 / frame_rate
		print("Completed converting video to frames at", frames_per_second, "frames per second")
		assert os.path.isdir(frames_dir)
		frame_paths = [join(frames_dir, im) for im in os.listdir(frames_dir)]
		dest_path = join(dest_path, get_file_name(vid_path) + "-seg" + get_file_extension(vid_path))
		apply_segmentation_dir(frame_paths, dest_path)
		print("Completed segmentation evaluation. Result is saved as", dest_path)
		frames_to_vid(frames_dir, dest_path, frames_per_second)
	
def frames_to_vid(frames_dir_path, dest_path, frame_rate):
	"""
	Converts provided frame images to a video and saves the video object to the given
	destination path.

	Parameters
	----------
	frames_dir_path:
		The path to the directory containing the frames that need to be used to construct
		the video
	dest_path:
		The path where the resulting video should be saved
	frame_rate:
		The frame rate that was used to break up the original video into frames	

	"""
	frame_arr = []
	frames = os.listdir(frames_dir_path)
	image_num = lambda x: float(get_file_name(x)[5:])
	frames.sort(key=image_num)
	for name in frames:
		frame_path = join(frames_dir_path, name)
		frame_image = cv2.imread(frame_path)
		height, width, layers = frame_image.shape
		size = (width, height)
		frame_arr.append(frame_image)
	out = cv2.VideoWriter(dest_path, cv2.VideoWriter_fourcc('m', 'p', '4','v'), frame_rate, size)
	for image in frame_arr:
		out.write(image)
	out.release()

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
	return os.path.splitext(os.path.split(file_path)[1])[0]

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
	return os.path.splitext(os.path.split(file_path)[1])[1]

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
		description='Process images using trained semantic segmentation model.',
		usage='semantic_segmentation.py [ --img | --vid | --dir ] [ image path I | video path V | directory path D ]')
	flag_group = parser.add_mutually_exclusive_group(required=True)
	flag_group.add_argument("--img", help="Use this flag when passing in an image file path I")
	flag_group.add_argument("--vid", help="Use this flag when passing in a video file path V")
	flag_group.add_argument("--dir", help="Use this flag when passing in a dir path containing images D")
	parser.add_argument("-r", help="Use this flag to specify the frame rate to convert the video to frame images",
						default=0.05)
	parser.add_argument("--mask", help="Use this flag to specify the Cityscapes mask image associated with the input image", 
						default=None)
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
	mask_path = input_args.mask
	rate = None
	if input_args.r:
		rate = float(input_args.r)
	process_input(img_path=image_path, dir_path=dir_path, vid_path=video_path, frame_rate=rate, mask_path=mask_path)

if __name__ == "__main__":
	main()
