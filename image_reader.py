from PIL import Image
import sys
import os
import copy

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