from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class BinarySegDataset(CustomDataset):
    """Binary semantic segmentation dataset.

    Labels are expected as single-channel PNG masks with:
    0 -> background
    1 -> object
    """

    CLASSES = ('background', 'object')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, img_suffix='.png', seg_map_suffix='.png', **kwargs):
        super(BinarySegDataset, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
