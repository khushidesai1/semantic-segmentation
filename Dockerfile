FROM pytorch:21.07-py3

COPY . .

RUN apt update && apt install python3.8 && apt install python3-pip
RUN apt install python3-opencv
RUN pip3 install torch torchvision
RUN pip install ninja tqdm


