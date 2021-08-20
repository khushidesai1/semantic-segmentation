FROM nvcr.io/nvidia/pytorch:21.07-py3

ENV DEBIAN_FRONTEND=noninteractive

COPY . .

RUN apt update && apt upgrade -y
RUN apt install -y python3-opencv
RUN pip3 install torch torchvision
RUN pip install --upgrade scikit-image
RUN pip install --upgrade imutils
RUN pip install ninja tqdm
RUN python setup.py
