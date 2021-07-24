import cv2
import os
from posixpath import join

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
	# image_num = lambda x: float(get_file_name(x)[5:])
	frames = sorted(frames)
	for name in frames:
		frame_path = join(frames_dir_path, name)
		frame_image = cv2.imread(frame_path)
		height, width, layers = frame_image.shape
		size = (width, height)
		frame_arr.append(frame_image)
	out = cv2.VideoWriter(dest_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), frame_rate, size)
	for image in frame_arr:
		out.write(image)
	out.release()

def frames_to_vid2():
	image_folder = frames_folder
	video_name = 'video-seg.avi'

	images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
	images = sorted(images)
	frame = cv2.imread(os.path.join(image_folder, images[0]))
	height, width, layers = frame.shape

	video = cv2.VideoWriter(video_name, 0, 20, (width,height))

	for image in images:
	    video.write(cv2.imread(os.path.join(image_folder, image)))

	cv2.destroyAllWindows()
	video.release()

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

frames_folder = '/home/khushi/Desktop/semantic-segmentation/runs/Oz7UU8VD'
fps = 30
#frames_to_vid(frames_folder, '~/Desktop/citys-video.mp4', fps)
frames_to_vid2()
