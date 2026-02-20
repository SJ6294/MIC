import os.path as osp

import mmcv

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class BinarySegDataset(CustomDataset):
    """Binary semantic segmentation dataset.

    Labels are expected as single-channel PNG masks with:
    0 -> background
    1 -> object

    If ``img_suffix`` is ``None`` (default), the dataset auto-detects one of
    the common image extensions in ``img_dir``.
    """

    CLASSES = ('background', 'object')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    COMMON_IMG_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    def __init__(self, img_suffix=None, seg_map_suffix='.png', **kwargs):
        if img_suffix is None:
            data_root = kwargs.get('data_root', None)
            img_dir = kwargs.get('img_dir', None)
            if img_dir is None:
                raise ValueError('img_dir must be provided for BinarySegDataset')
            if data_root is not None and not osp.isabs(img_dir):
                img_path = osp.join(data_root, img_dir)
            else:
                img_path = img_dir
            img_suffix = self._infer_img_suffix(img_path)

        super(BinarySegDataset, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

    def _infer_img_suffix(self, img_path):
        for suffix in self.COMMON_IMG_SUFFIXES:
            if any(mmcv.scandir(img_path, suffix, recursive=True)):
                return suffix
        raise FileNotFoundError(
            f'No image files found in {img_path} with supported suffixes '
            f'{self.COMMON_IMG_SUFFIXES}. Please set img_suffix explicitly.')
