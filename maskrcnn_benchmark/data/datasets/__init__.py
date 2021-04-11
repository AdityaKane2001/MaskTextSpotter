# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .coco import COCODataset
from .concat_dataset import ConcatDataset,MixDataset
from .icdar import IcdarDataset
from .synthtext import SynthtextDataset
from .scut import ScutDataset
from .total_text import TotaltextDataset
from .answersheet import AnswersheetDataset
__all__ = ["AnswersheetDataset","COCODataset", "ConcatDataset","IcdarDataset","SynthtextDataset","MixDataset","ScutDataset","TotaltextDataset"]
