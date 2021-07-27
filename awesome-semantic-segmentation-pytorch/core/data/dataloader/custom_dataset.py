"""Prepare custom data for custom data evaluation"""
import os
import numpy as np
from posixpath import join

from PIL import Image
from .segbase import SegmentationDataset


class CustomDatasetSegmentation(SegmentationDataset):
    """Custom Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to dataset folder. Default is ''
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = CitySegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    NUM_CLASS = 19

    def __init__(self, custom_dataset=None, root='', split='train', mode='testval', transform=None, **kwargs):
        super(CustomDatasetSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        images_folder = custom_dataset
        self.images = [join(images_folder, path) for path in os.listdir(images_folder)]
        assert len(self.images) > 0
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                              23, 24, 25, 26, 27, 28, 31, 32, 33]
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        assert self.mode == 'testval'
        
        img = self._img_transform(img)
        
        if self.transform is not None:
            img = self.transform(img)
        return img, os.path.basename(self.images[index])
    
    def __len__(self):
    	return len(self.images)

    @property
    def pred_offset(self):
        return 0

if __name__ == '__main__':
    dataset = CustomDatasetSegmentation()
