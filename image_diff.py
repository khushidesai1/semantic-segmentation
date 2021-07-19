from posixpath import join
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import cv2
import os
from semantic_segmentation import get_file_name

def parse_args():
	"""
	Builds an argument parser that creates 2 flags. The first flag takes
	in the path to the first image for comparison and the second flag takes
	in the path to the second image for comparison.

	Returns
	-------
	ArgumentParser
		The object containing all the specifications for the arguments
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--first", required=True, help="first input image")
	parser.add_argument("-s", "--second", required=True, help="second input image")
	return parser.parse_args()

def create_diff_image(pathA, pathB):
	"""
	Takes in the path to the first and second image, creates
	a differential image and a threshold image and stores the two
	images in the folder ./diffs/[image1 name]-[image2 name]/. Prints the
	SSIM and the difference score that was calculated while creating the
	differential image.

	Parameters
	----------
	pathA: str
		The path to the first image to compare
	pathB: str
		The path to the second image to compare
	  
	"""
	imageA = cv2.imread(pathA)
	imageB = cv2.imread(pathB)
	
	grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
	grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
	
	diff = calculate_diff(grayA, grayB)

	thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	cnts = get_contours(thresh.copy())

	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
		cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
	
	nameA = get_file_name(pathA)
	nameB = get_file_name(pathB)
	dest_path = join('./diffs/', nameA + "-" + nameB)
	write_images(diff, thresh, dest_path)

def calculate_diff(grayA, grayB):
	"""
	Takes in 2 grayscale images and calculates the difference between the 
	two images. Returns the diff image and prints the SSIM score along with the 
	difference score.

	Parameters
	----------
	grayA: Image object
		The first grayscale image to compare
	grayB: Image object
		The second grayscale image to compare
	
	Returns
	-------
	Image
		The differential pixel image that will be used to create the differential image
	"""
	(score, diff) = compare_ssim(grayA, grayB, full=True)
	diff = (diff * 255).astype("uint8")
	print("SSIM: {}".format(score))
	print('Difference:', 1 - score)
	return diff

def get_contours(thresh):
	"""
	Takes in the differential matrix pixel and returns the 
	countours for the differential image. 

	Parameters
	-----------
	thresh: Image
		The threshold pixel matrix corresponding to the comparison of the two images
	
	Returns
	-------
	Contour
		The contours needed to create the image using the differential
	"""
	cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	return cnts

def write_images(diff_image, thresh_image, dest_path):
	"""
	Takes in the differential image and the threshold image and saves the images
	within the given destination path.

	Parameters
	----------
	diff_image: Image
		The differential image to be saved
	thresh_image: Image
		The threshold image to be saved
	dest_path: str
		The path to the directory in which the differential and threshold images will be
		saved
	"""
	if not os.path.isdir(dest_path):
		os.makedirs(dest_path)
	diff_path = join(dest_path, 'image-diff.png')
	thresh_path = join(dest_path, 'image-thresh.png')
	cv2.imwrite(diff_path, diff_image)
	cv2.imwrite(thresh_path, thresh_image)

if __name__ == '__main__':
	args = parse_args()
	create_diff_image(args.first, args.second)
