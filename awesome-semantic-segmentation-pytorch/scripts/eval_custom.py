import os
import sys

from torchvision import transforms

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from core.utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler
from core.utils.logger import setup_logger
from core.models.model_zoo import get_segmentation_model
from core.data.dataloader import get_segmentation_dataset
from core.utils.score import SegmentationMetric

import argparse

class CustomEvaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                            aux=args.aux, pretrained=True, pretrained_base=False,
                                            norm_layer=BatchNorm2d).to(self.device)
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank)
        self.model.to(self.device)

        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='testval', transform=input_transform)
        self.metric = SegmentationMetric(val_dataset.num_class)

    def eval(self):
        pass

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluator that evaluates using a model using custom images and ground truth images as input',
        usage='eval_custom.py [--input_image I]')
    parser.add_argument('--model', type=str, default='fcn',
                        choices=['fcn32s', 'fcn16s', 'fcn8s',
                                 'fcn', 'psp', 'deeplabv3', 'deeplabv3_plus',
                                 'danet', 'denseaspp', 'bisenet',
                                 'encnet', 'dunet', 'icnet',
                                 'enet', 'ocnet', 'ccnet', 'psanet',
                                 'cgnet', 'espnet', 'lednet', 'dfanet'],
                        help='model name (default: fcn32s)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet18', 'resnet50',
                                 'resnet101', 'resnet152', 'densenet121',
                                 'densenet161', 'densenet169', 'densenet201'],
                        help='backbone name (default: vgg16)')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        choices=['pascal_voc', 'pascal_aug', 'ade20k',
                                 'citys', 'sbu'],
                        help='dataset name (default: pascal_voc)')
    parser.add_argument('--input-image', type='str', help='directory to the input image', required=True)
    parser.add_argument('--input-gt', type='str', help='directory to the input image ground truth', required=True)
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    return parser.parse_args()

def main():
    pass
    args = parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    args.save_pred = True
    if args.save_pred:
        outdir = '../runs/custom_pred_pic/{}_{}_{}'.format(args.model, args.backbone, args.dataset)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(),
                          filename='{}_{}_{}_log.txt'.format(args.model, args.backbone, args.dataset), mode='a+')
    evaluator = CustomEvaluator(args)
    evaluator.eval()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()