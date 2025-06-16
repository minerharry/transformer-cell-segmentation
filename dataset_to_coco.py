

import datetime
import json
import os
from pathlib import Path
from random import shuffle,sample
import shutil
from typing import Iterable, Sequence
from PIL import Image
from imageio.v3 import imread
import numpy as np
from pycococreatortools import pycococreatortools as pyco
from tqdm import tqdm
from read_dataset import shared_files
from skimage.measure import label
from pycocotools.coco import COCO


    

def create_video_info(video_id:int,size:tuple[int,int],length:int,file_names:list[str]):
    return {
        "id":video_id,
        "width":size[0],
        "height":size[1],
        "length":length,
        "file_names":file_names
    }

def create_video_annotation_info(binary_masks:Iterable[np.ndarray],video_id:int,annotation_id:int,category_id:int,iscrowd:bool,allow_missing:bool=False,):
    mask_annotations = [pyco.create_annotation_info(0,0,{"id":0,"is_crowd":iscrowd},binary_mask) for binary_mask in binary_masks ]

    if not any(mask_annotations):
        return None
    else:
        templ = [m for m in mask_annotations if m is not None][0]
        if any(map(lambda x: x is None,mask_annotations)):
            if not allow_missing:
                return None
            # print(mask_annotations)
            tempseg = templ["segmentation"]
            if isinstance(tempseg,dict):
                size = templ["segmentation"]["size"]
                counts = [size[0]*size[1]]
                empty_seg = {"counts":counts,"size":size}
            else:
                empty_seg = [[]]
        else:
            empty_seg = {}

        mask_segmentations = [m["segmentation"] if m is not None else empty_seg for m in mask_annotations]
        mask_areas = [m["area"] if m is not None else 0 for m in mask_annotations]
        mask_bboxes = [m["bbox"] if m is not None else [] for m in mask_annotations]

    return {
        "height": templ["height"],
        "width":templ["width"],
        "length":1, #for some reason
        "areas":mask_areas,
        "category_id":category_id,
        "iscrowd":iscrowd,
        "id":annotation_id,
        "video_id":video_id,
        "bboxes":mask_bboxes,
        "segmentations": mask_segmentations,
    }


def make_video_inference_dataset(image_groups:Sequence[Sequence[str|Path]],
                 output_data_folder:str|Path,
                 images_subfolder_name:str="train2017",
                #  val_images_subfolder_name:str="val2017",
                 json_file_path:str="instances_train2017.json",
                 name:str="dataset",
                 desc=None):
    """
    final directory structure is
    output_data_folder/
        train_images_subfolder_name/
            train_image_1
            train_image_2
            ...
        val_images_subfolder_name/
            val_image_1
            val_image_2
            ...
        train_annotations_file_path
        val_annotations_file_path
    """


    if desc is None:
        desc = name

    INFO = {
        "description": f"{desc}",
        "url": "https://github.com/minerharry",
        "version": "0.1.0",
        "year": 2025,
        "contributor": "minerharry",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]

    CYTO_ID = 1

    CATEGORIES = [
        {
            'id': CYTO_ID,
            'name': 'cytplasm',
        },
    ]

    coco_output = ({
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "videos": [],
        "annotations": []
    })


    # val_cutoff = train_split*num_videos

    out_impath = Path(output_data_folder)/images_subfolder_name
    #     Path(output_data_folder)/val_images_subfolder_name
    # )

    out_impath.mkdir(parents=True,exist_ok=True)


    valmode = False
    nanns = 0
    for video_id,(imgroup,) in enumerate(tqdm(list(zip(image_groups)),desc="Making Dataset",leave=False)):
        
        im0 = Image.open(imgroup[0])

        for impath in imgroup:
            shutil.copy(impath,out_impath);

        vid_info = create_video_info(
                video_id, im0.size, len(imgroup), [Path(i).name for i in imgroup])
        
        coco_output["videos"].append(vid_info)

    # for output,jsonpath in zip(coco_outputs,[train_annotations_file_path,val_annotations_file_path]):
    #     # if len(output["annotations"]) == 0:
    #     #     continue
    jsonpath = Path(output_data_folder)/json_file_path
    jsonpath.parent.mkdir(parents=True,exist_ok=True);

    with open(jsonpath,"w") as f:
        json.dump(coco_output,f)

    return coco_output

def make_video_dataset(image_groups:list[list[str|Path]],mask_groups:list[list[str|Path]],
                 output_data_folder:str|Path,
                 images_subfolder_name:str="train2017",
                #  val_images_subfolder_name:str="val2017",
                 annotations_file_path:str="instances_train2017.json",
                #  annotations_file_path:str="instances_val2017.json",
                 prelabeled=True,
                #  train_split:float=1.0,
                 name:str="dataset",
                 desc=None):
    """
    final directory structure is
    output_data_folder/
        train_images_subfolder_name/
            train_image_1
            train_image_2
            ...
        val_images_subfolder_name/
            val_image_1
            val_image_2
            ...
        train_annotations_file_path
        val_annotations_file_path
    """


    if desc is None:
        desc = name

    INFO = {
        "description": f"{desc}",
        "url": "https://github.com/minerharry",
        "version": "0.1.0",
        "year": 2025,
        "contributor": "minerharry",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]

    CYTO_ID = 1

    CATEGORIES = [
        {
            'id': CYTO_ID,
            'name': 'cytplasm',
        },
    ]

    coco_output = ({
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "videos": [],
        "annotations": []
    })


    assert len(image_groups) == len(mask_groups)

    num_videos = len(image_groups)

    # val_cutoff = train_split*num_videos
    # from IPython import embed; embed()

    image_groups,mask_groups = zip(*sample(list(zip(image_groups,mask_groups)),num_videos)) #pretty sure this works

    out_impath = Path(output_data_folder)/images_subfolder_name
    #     Path(output_data_folder)/val_images_subfolder_name
    # )

    out_impath.mkdir(parents=True,exist_ok=True)


    valmode = False
    nanns = 0
    for video_id,(imgroup,maskgroup) in enumerate(tqdm(list(zip(image_groups,mask_groups)),desc="Making Dataset",leave=False)):
        
        assert len(imgroup) == len(maskgroup), "Image and mask sequence lengths must match"

        im0 = Image.open(imgroup[0])

        for impath in imgroup:
            shutil.copy(impath,out_impath);

        vid_info = create_video_info(
                video_id, im0.size, len(imgroup), [Path(i).name for i in imgroup])
        
        coco_output["videos"].append(vid_info)

        labeled = np.array([imread(mask) for mask in maskgroup])
        assert prelabeled == True

        for num in np.unique(labeled):
            if num == 0:
                continue
            mask = labeled == num
            annotation_info = create_video_annotation_info(
                mask, video_id, nanns, CYTO_ID, False)
            
            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)
                nanns += 1

    # for output,jsonpath in zip(coco_outputs,[train_annotations_file_path,val_annotations_file_path]):
    #     # if len(output["annotations"]) == 0:
    #     #     continue
    jsonpath = Path(output_data_folder)/annotations_file_path
    jsonpath.parent.mkdir(parents=True,exist_ok=True);

    with open(jsonpath,"w") as f:
        json.dump(coco_output,f)

    return coco_output






def make_dataset(input_images:str|Path,input_masks:str|Path,
                 output_data_folder:str|Path,
                 train_images_subfolder_name:str="train2017",
                 val_images_subfolder_name:str="val2017",
                 train_annotations_file_path:str="instances_train2017.json",
                 val_annotations_file_path:str="instances_val2017.json",
                 prelabeled=False,
                 train_split:float=1.0,
                 name:str="dataset",
                 desc=None):
    """
    final directory structure is
    output_data_folder/
        train_images_subfolder_name/
            train_image_1
            train_image_2
            ...
        val_images_subfolder_name/
            val_image_1
            val_image_2
            ...
        train_annotations_file_path
        val_annotations_file_path
    """


    if desc is None:
        desc = name

    INFO = {
        "description": f"{desc}",
        "url": "https://github.com/minerharry",
        "version": "0.1.0",
        "year": 2025,
        "contributor": "minerharry",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]

    CYTO_ID = 1

    CATEGORIES = [
        {
            'id': CYTO_ID,
            'name': 'cytplasm',
        },
    ]

    coco_outputs = ({
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    },
    {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    })

    names = shared_files(os.listdir(input_images),os.listdir(input_masks))

    val_cutoff = train_split*len(names)

    shuffle(names)

    out_impaths = (
        Path(output_data_folder)/train_images_subfolder_name,
        Path(output_data_folder)/val_images_subfolder_name
    )

    [p.mkdir(parents=True,exist_ok=True) for p in out_impaths]


    valmode = False
    nanns = 0
    for image_id,imname in enumerate(tqdm(names,desc="Making Dataset",leave=False)):
        impath = Path(input_images)/imname
        maskpath = Path(input_masks)/imname

        if image_id > val_cutoff:
            valmode = True

        coco_output = coco_outputs[valmode]
        out_impath = out_impaths[valmode]

        shutil.copy(impath,out_impath);

        image = Image.open(impath)
        image_info = pyco.create_image_info(
                image_id, str(imname), image.size)
        
        coco_output["images"].append(image_info)

        category_info = {'id': CYTO_ID, 'is_crowd': True}
        binary_mask = (imread(maskpath)).astype(np.uint8)

        assert prelabeled == True,"Cannot automatically join labels accross frame! If need be, please label and run through the tracking algorithm!"
        labeled = label(binary_mask > 0) if not prelabeled else binary_mask

        for num in np.unique(labeled):
            mask = labeled == num
            annotation_info = pyco.create_annotation_info(
                nanns, image_id, category_info, mask,
                image.size, tolerance=2)
            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)
                nanns += 1

    for output,jsonpath in zip(coco_outputs,[train_annotations_file_path,val_annotations_file_path]):
        # if len(output["annotations"]) == 0:
        #     continue
        jsonpath = Path(output_data_folder)/jsonpath
        jsonpath.parent.mkdir(parents=True,exist_ok=True);

        with open(jsonpath,"w") as f:
            json.dump(output,f)

    return coco_outputs

if __name__ == "__main__":
    make_dataset("datasets/optotaxis/images","datasets/optotaxis/masks","coco_test")