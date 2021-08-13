FROM nvcr.io/nvidia/pytorch:21.07-py3

ENV DEBIAN_FRONTEND=noninteractive

COPY . .

RUN apt update
RUN apt install -y python3-opencv
RUN pip3 install torch torchvision
RUN pip install ninja tqdm
RUN python setup.py