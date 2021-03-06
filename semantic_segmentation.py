from posixpath import join
import os
import argparse
import cv2
import random
import string
import re

MODEL = ' --model '
BACKBONE = ' --backbone '
DATASET = ' --dataset citys'

def apply_segmentation_dir(model, backbone, dir_path, dest_path, ngpus=1):
	"""
	Takes in image paths from a directory and feeds each image into the trained Tramac neural 
	network model. Writes the resulting overlays for each image to a specified directory.
	
	Parameters
	----------
	model: str
		The model to use to perform evaluation
	backbone: str
		The backbone to use to perform evaluation
	dir_path: str
		The path to the directory containing the images to be segmented
	dest_path: str
		The path to the destination directory where the segmented images will be stored
	ngpus: int, optional
		The number of GPUs the user wants to utilize for evaluation
	"""
	os.chdir('./awesome-semantic-segmentation-pytorch/scripts')
	if not os.path.isdir(dir_path):
		dir_path = join('../../', dir_path)
	dest_path = join('../../', dest_path)
	outdir = ' --outdir ' + dest_path
	input_folder = ' --custom-dataset ' + dir_path
	parameters = MODEL + model + BACKBONE + backbone + DATASET + input_folder + outdir
	if ngpus > 1:
		os.system('python -m torch.distributed.launch --nproc_per_node=' + str(ngpus) + '  eval_custom_dataset.py' + parameters)
	else:
		os.system('python eval_custom_dataset.py' + parameters)
	os.chdir('../..')

def apply_single_segmentation(model, backbone, img_path, dest_path, mask_path=None, ngpus=1):
	"""
	Takes in a single image path and feeds the image into the trained Tramac neural network model.
	Writes the resulting overlay image to the specified directory path. 

	Parameters
	----------
	model: str
		The model to use to perform evaluation
	backbone: str
		The model to use to perform evaluation
	img_path: str
		The path to the given image
	dest_path: str
		The path to the destination directory where the segmented image will be stored
	mask_path: str
		The path to the mask image for performing metric evaluation
	ngpus: int, optional
		The number of GPUs the user wants to utilize for evaluation
	"""
	os.chdir('./awesome-semantic-segmentation-pytorch/scripts')
	if not os.path.isfile(img_path):
		img_path = join('../../', img_path)
	dest_path = join('../../', dest_path)
	img = ' --input-pic ' + img_path
	outdir = ' --outdir ' + dest_path
	parameters = MODEL + model + BACKBONE + backbone + DATASET + img + outdir
	if ngpus > 1:
		if not mask_path:
			os.system('python -m torch.distributed.launch --nproc_per_node=' + str(ngpus) + ' eval_custom.py' + parameters)
		else:
			mask = ' --input-gt ' + mask_path
			mask_parameters = MODEL + model + BACKBONE + backbone + DATASET + img + outdir + mask
			os.system('python -m torch.distributed.launch --nproc_per_node=' + str(ngpus) + ' eval_custom_metric.py' + mask_parameters)
	else:
		if not mask_path:
			os.system('python eval_custom.py' + parameters)
		else:
			mask = ' --input-gt ' + mask_path
			mask_parameters = MODEL + model + BACKBONE + backbone + DATASET + img + outdir + mask
			os.system('python eval_custom_metric.py' + mask_parameters)
	os.chdir('../..')
					
def process_input(model='psp', backbone='resnet50', img_path=None, dir_path=None, 
				vid_path=None, frame_rate=0.5, mask_path=None, ngpus=1):
	"""
	Reads the given image, images in the given directory, or image frames from the given video.
	Based on the format of the input, processes the image(s) by applying segmentation to each
	video and saving the results in a directory.

	The results will be stored in the following formats:
	- For a single image/directory of images: ./runs/[random string code]/[image name]-seg.jpg
	- For a video: ./runs/[random string code]/[video name]-seg.mp4

	Parameters
	----------
	model: str, optional
		The model to perform evaluation, with the default being PSPNet
	backbone: str, optional
		The backbone to perform evaluation, with the default being ResNet50
	img_path: str, optional
		The path to the given image
	dir_path: str, optional
		The path to the given directory
	vid_path: str, optional
		The path to the given video
	frame_rate: double
		The desired frame rate to conver the video to frame images
	mask_path: str, optional
		The path to the mask image for performing metric evaluation
	ngpus: int, optional
		The number of GPUs the user wants to utilize for evaluation with the default as 1
	"""
	run_id = generate_id()
	dest_path = join("./runs", run_id)
	if not os.path.isdir(dest_path):
		os.makedirs(dest_path)
	if img_path: 
		assert os.path.isfile(img_path)
		print("Completed reading image:", img_path)
		
		dest_path = join(dest_path, get_file_name(img_path) + "-seg" + get_file_extension(img_path))
		apply_single_segmentation(model, backbone, img_path, dest_path, mask_path, ngpus)
		
		print("Completed segmentation evaluation. Result is saved as", dest_path)
	if dir_path:
		assert os.path.isdir(dir_path)
		print("Completed reading images from directory:", dir_path)

		apply_segmentation_dir(model, backbone, dir_path, dest_path, ngpus)
		
		print("Completed segmentation evaluation. Result is saved in", dest_path)
	if vid_path:
		frames_dir = join(dest_path, get_file_name(vid_path) + '-frames')
		vid_to_frames(vid_path, frames_dir, 1 / frame_rate)
		
		print("Completed converting video to frames at", frame_rate, "frames per second")
		assert os.path.isdir(frames_dir)
		
		segmented_frames_dir = join(dest_path, get_file_name(vid_path) + '-seg-frames')
		if not os.path.isdir(segmented_frames_dir):
			os.makedirs(segmented_frames_dir)
		dest_path = join(dest_path, get_file_name(vid_path) + "-seg" + get_file_extension(vid_path))
		
		apply_segmentation_dir(model, backbone, join('../../',frames_dir), segmented_frames_dir, ngpus)
		
		frames_to_vid(segmented_frames_dir, dest_path, frame_rate)
		print("Completed segmentation evaluation. Result is saved as", dest_path)
	
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
	images = [img for img in os.listdir(frames_dir_path) if img.endswith(".png")]
	image_num = lambda x: int(re.findall(r"[\d]+", x)[0])
	images = sorted(images, key=image_num)
	frame = cv2.imread(os.path.join(frames_dir_path, images[0]))
	height, width, layers = frame.shape

	video = cv2.VideoWriter(dest_path, 0, frame_rate, (width,height))

	for image in images:
	    video.write(cv2.imread(os.path.join(frames_dir_path, image)))

	cv2.destroyAllWindows()
	video.release()

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
	num = 0
	success = get_frame(vidcap, sec, num, frames_dir)
	while success:
		sec = round(sec + frame_rate, 2)
		success = get_frame(vidcap, sec, num, frames_dir)
		num += 1
	return frames_dir

def get_frame(vidcap, sec, num, frames_dir):
	"""
	Gets a single frame using the VideoCapture object and saves the frame image to the
	directory ./[video name]-frames/

	Parameters
	----------
	vidcap: VideoCapture object
		The VideoCapture object read from the video file path
	sec: double
		The second value of the video to capture
	num: int
		The frame number
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
		cv2.imwrite(frames_dir + "/image" + str(num) + ".png", img)
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
	Builds a parser with flags to accept a single image, a dir of images and a video file.
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
	flag_group.add_argument('--img', help='Use this flag when passing in an image file path I')
	flag_group.add_argument('--vid', help='Use this flag when passing in a video file path V')
	flag_group.add_argument('--dir', help='Use this flag when passing in a dir path containing images D')
	parser.add_argument('-r', help='Use this flag to specify the frame rate to convert the video to frame images',
						default=20)
	parser.add_argument('--mask', help="Use this flag to specify the Cityscapes mask image associated with the input image", 
						default=None)
	parser.add_argument("--ngpus", help="Use this flag to specify how many GPUs you would like the system to utilize",
						default=1)
	parser.add_argument("--model", help='Use this flag to specify a model to use for evaluation other than the default PSPNet',
						default='psp')
	parser.add_argument("--backbone", help='Use this flag to specify a backbone to use for evaluation other than the default PSPNet',
						default='resnet50')
	return parser.parse_args()

def main():
	"""
	Processes the arguments, reads the given image, images in the directory or the given video,
	passes the images into the trained semantic segmentation model, and writes the output to a directory.
	"""
	args = parse_args()
	image_path = args.img
	video_path = args.vid
	dir_path = args.dir
	mask_path = args.mask
	model = args.model
	backbone = args.backbone
	gpus = None
	rate = None
	if args.r:
		rate = float(args.r)
	if args.ngpus:
		gpus = int(args.ngpus)
	process_input(model=model, backbone=backbone, img_path=image_path, dir_path=dir_path, vid_path=video_path, frame_rate=rate, mask_path=mask_path, ngpus=gpus)

if __name__ == "__main__":
	main()
