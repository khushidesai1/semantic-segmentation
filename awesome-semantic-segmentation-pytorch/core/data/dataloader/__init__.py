"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .mscoco import COCOSegmentation
from .cityscapes import CitySegmentation
from .custom_metric import CustomMetricSegmentation
from .custom_dataset import CustomDatasetSegmentation
from .custom import CustomSegmentation
from .ade import ADE20KSegmentation
from .pascal_voc import VOCSegmentation
from .pascal_aug import VOCAugSegmentation
from .sbu_shadow import SBUSegmentation

datasets = {
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'coco': COCOSegmentation,
    'citys': CitySegmentation,
    'custom-metric': CustomMetricSegmentation,
    'custom': CustomSegmentation,
    'custom-dataset': CustomDatasetSegmentation,
    'sbu': SBUSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
