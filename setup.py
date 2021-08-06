from posixpath import join
import os

if __name__ == "__main__":
    os.chdir('./awesome-semantic-segmentation-pytorch/core/nn')
    os.system('python setup.py build develop')