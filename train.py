import os
import argparse

MODEL = ' --model '
BACKBONE = ' --backbone '
DATASET = ' --dataset citys'
LR = '--lr '
EPOCHS = '--epochs '

def train(model, backbone, epochs=100, loss_rate=0.0001, ngpus=1):
    """
    Function that uses the inputted parameters to call training using the Tramac segmentation repository.
    Trains a model to perform semantic segmentation on any given inputted image.

    Parameters
    ----------
    model: str
        The model to use for training
    backbone: str
        The backbone to use for training
    epochs: int
        The number to epochs to run for training
    loss_rate: float
        The loss rate to use for training
    ngpus: int
        The number of GPUs to use for training
    """
    os.chdir('./awesome-semantic-segmentation-pytorch/scripts')
    parameters = MODEL + model + BACKBONE + backbone + DATASET + LR + loss_rate + EPOCHS + epochs
    if ngpus > 1:
        os.system('export NGPUS=' + str(ngpus))
        os.system('python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py' + parameters)
    else:
        os.system('python train.py' + parameters)
    os.chdir('../../')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a model to perform semantic segmentation.',
        usage='train.py --model [ model M ] --backbone [ backbone B ] --lr [ loss rate LR ] --epochs [ epochs E ] --ngpus [ NGPUS ]'
    )
    parser.add_argument('--ngpus', help='Use this flag to specify how many GPUs you would like the system to utilize', default=1)
    parser.add_argument('--model', help='Use this flag to specify a model to use for evaluation other than the default PSPNet', default='psp')
    parser.add_argument('--backbone', help='Use this flag to specify a backbone to use for evaluation other than the default PSPNet', default='resnet50')
    parser.add_argument('--lr', help='Use this flag to specify the loss rate while training the model', default=0.0001)
    parser.add_argument('--epochs', help='Use this flag to specify the number of epochs to use for training', default=100)

def main():
    """
    Processes the arguments and calls the train function using the given parameters.
    """
    args = parse_args()
    model = args.model
    backbone = args.backbone
    ngpus = args.ngpus
    loss_rate = args.lr
    epochs = args.epochs
    train(model=model, backbone=backbone, ngpus=ngpus, loss_rate=loss_rate, epochs=epochs)

if __name__ == "__main__":
	main()
