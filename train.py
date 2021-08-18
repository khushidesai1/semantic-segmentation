import os
import argparse

MODEL = ' --model '
BACKBONE = ' --backbone '
DATASET = ' --dataset citys'
LR = ' --lr '
EPOCHS = ' --epochs '

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
    parameters = MODEL + model + BACKBONE + backbone + DATASET + LR + str(loss_rate) + EPOCHS + str(epochs)
    if ngpus > 1:
        os.system('python -m torch.distributed.launch --nproc_per_node=' + str(ngpus) + ' train.py' + parameters)
    else:
        os.system('python train.py' + parameters)
    os.chdir('../../')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a model to perform semantic segmentation.',
        usage='train.py --model [ model M ] --backbone [ backbone B ] --lr [ loss rate LR ] --epochs [ epochs E ] --ngpus [ NGPUS ]'
    )
    parser.add_argument('--ngpus', help='Use this flag to specify how many GPUs you would like the system to utilize', default=1)
    parser.add_argument('--model', required=True, help='Use this flag to specify a model to use for evaluation other than the default PSPNet', default='psp')
    parser.add_argument('--backbone', required=True, help='Use this flag to specify a backbone to use for evaluation other than the default PSPNet', default='resnet50')
    parser.add_argument('--lr', help='Use this flag to specify the loss rate while training the model', default=0.0001)
    parser.add_argument('--epochs', help='Use this flag to specify the number of epochs to use for training', default=100)
    return parser.parse_args()

def main():
    """
    Processes the arguments and calls the train function using the given parameters.
    """
    args = parse_args()
    model = str(args.model)
    backbone = str(args.backbone)
    ngpus = int(args.ngpus)
    loss_rate = float(args.lr)
    epochs = int(args.epochs)
    train(model=model, backbone=backbone, ngpus=ngpus, loss_rate=loss_rate, epochs=epochs)

if __name__ == "__main__":
	main()
