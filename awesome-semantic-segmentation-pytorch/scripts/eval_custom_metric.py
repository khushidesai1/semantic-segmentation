import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torch.nn as nn

from core.utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler
from core.utils.logger import setup_logger
from core.models.model_zoo import get_segmentation_model
from core.data.dataloader import get_segmentation_dataset
from core.utils.visualize import get_color_pallete
from core.utils.score import SegmentationMetric

from train import parse_args

class CustomMetricEvaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        dataset = get_segmentation_dataset('custom-metric', input_pic=args.input_pic, input_gt=args.input_gt, mode='testval', transform=input_transform, split='val')
        sampler = make_data_sampler(dataset, False, args.distributed)
        batch_sampler = make_batch_data_sampler(sampler, images_per_batch=1)
        self.dataloader = data.DataLoader(dataset=dataset, batch_sampler=batch_sampler, num_workers=args.workers, pin_memory=True)
        
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                            aux=args.aux, pretrained=True, pretrained_base=False,
                                            norm_layer=BatchNorm2d).to(self.device)
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank)
        self.model.to(self.device)

        self.metric = SegmentationMetric(dataset.num_class)

    def eval(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        logger.info('Starting Evaluation')
        for i, (image, target) in enumerate(self.dataloader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                    outputs = model(image)
        
            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            logger.info("PixAcc: {:.4f}, mIoU: {:.4f}".format(pixAcc * 100, mIoU * 100))
            if self.args.save_pred:
                pred = torch.argmax(outputs[0], 1)
                pred = pred.cpu().data.numpy()

                predict = pred.squeeze(0)
                mask = get_color_pallete(predict, self.args.dataset)
                mask.save(outpath)
        synchronize()
        logger.info('Evaluation Completed')

if __name__ == '__main__':
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
        outpath = args.outdir
        outdir = os.path.dirname(outpath)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(),
                          filename='{}_{}_{}_log.txt'.format(args.model, args.backbone, args.dataset), mode='a+')
    evaluator = CustomMetricEvaluator(args)
    evaluator.eval()
    torch.cuda.empty_cache()
