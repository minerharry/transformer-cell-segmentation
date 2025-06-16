


from pathlib import Path
from pycocotools.coco import COCO

def calculate_mAP_coco(gt_dataset_annfile:str|Path,detections_json:str|Path):