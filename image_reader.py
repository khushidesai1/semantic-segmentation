from PIL import Image
import sys
import os
import copy

def main():
	# process arguments
	# read images
	# process
	# write output

if __name__ == "__main__":
	main()

	
def process_args(sys.argv):
	


# Takes in either an image path, folder path or a video path.
# Given an image path, the function reads the image and returns it as an image object.
# Given a folder path, the function reads the images in the folder and returns a list of image objects.
# Given a video path, the function turns the videos into frames of images and returns a list of image objects.
def read_images(img_path=None, folder_path=None, vid_path=None):
	if image_path:
		img = validate_img(image_path)
		print("Completed reading image:", image_path)
		return img
	if folder_path:
		img_paths = os.listdir(folder_path)
		img_list = []
		for img_name in img_paths:
			img_list.append(copy.deepcopy(validate_img(folder_path + "/" + img_name)))
		print("Completed reading images from folder:", folder_path)
		return img_list
	if video_path:
		# process video into frames

# Takes in an image filepath and checks whether it is a valid image or not. If it is, the function returns
# the image object, otherwise it throws an error.
def validate_img(filepath):
	try:
		img = Image.open(filepath)
		return img
	except:
		print("Value Error:", folder_path + "/" + img_name, "is not a valid image file")
		sys.exit(1)
	
if len(sys.argv) <= 1:
    print("Argument Error: no path provided to read images")
    sys.exit(1)

path = sys.argv[1]
try:
    file_list = os.listdir(path)
except:
    print("Value Error: please provide a valid directory")
    sys.exit(1)
image_list = []

for filename in file_list:
    try:
        image_list.append(copy.deepcopy(Image.open(path + "/" + filename)))
    except:
        print("Value Error:", path + "/" + filename, "is not a valid image file")
        sys.exit(1)

print("Completed reading all images in", path)
