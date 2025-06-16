import shutil
from typing import Iterable

import numpy as np
import torch
from segmentation_training import LoadableModel
import fiftyone as fo
from fiftyone.core.sample import Sample
from pathlib import Path
import os
from get_adjacency_videos import get_video_groups
from segmentation_training import optotaxis_enum_regkey
from dataset_to_coco import make_video_dataset, make_video_inference_dataset
import sys

sys.path.append(r"C:\Users\miner\Documents\Github\SeqFormer")
from train_custom import train_custom
from predict_custom import predict_custom

class SeqFormerModel(LoadableModel):
    def __init__(self,modelpath:Path|str|os.PathLike[str],gpu:bool=False,video_length:int|None=3,regkey:tuple[str,str,int]|None=None,**kwargs):
        self.vidlength = video_length
        self.pretrain_loc = modelpath;
        self.save_path = None
        if regkey is None:
            raise ValueError("Must provide regex key for filename adjacency as the `regkey` kwarg, tupel of (filename_regex, filename_format_regex, idx_key)")
        self.regkey = regkey

    
    def fine_tune(self,dataset:fo.Dataset,batch_size:int=2,epochs:int=100,**kwargs)->"LoadableModel":
        groups:list[list[Sample]] = get_video_groups(
            dataset,
            self.vidlength,
            filename_key=lambda sample: os.path.basename(sample.filepath),
            filename_regex=self.regkey[0],
            format_regex=self.regkey[1],
            idx_key=self.regkey[2],
        )

        from IPython import embed; embed()

        impaths = [[samp.filepath for samp in g] for g in groups]
        maskpaths = [[samp.ground_truth.mask_path for samp in g] for g in groups]

        dataset_path = f"training/video_dataset_{id(self)}"
        
        make_video_dataset(impaths,maskpaths,dataset_path,images_subfolder_name="train/JPEGImages",annotations_file_path="annotations/instances_train_sub.json",prelabeled=True)

        if self.save_path is None:
            self.save_path = f"seqformer_saves/{id(self)}"

        train_custom(epochs,self.vidlength or len(impaths[0]),self.save_path,"YoutubeVIS",ytvis_path=dataset_path,pretrain_weights=str(self.pretrain_loc))

        return self

    def save_model(self,savepath:str|Path|None=None):
        if not savepath:
            return #the model already autosaves to its own savepath so no need to do anything
        
        if self.save_path and savepath:
            file = Path(self.save_path)/"checkpoint.pth"

            dest = Path(savepath)/"checkpoint.pth"
            shutil.copyfile(file,dest)
        
        self.save_path = savepath

    def predict(self,x:np.ndarray,verbose:bool=True)->torch.Tensor:
        raise ValueError("numpy array prediction not supported. please use predict_dataset instead")
    
    def predict_dataset(self,in_folder:str|Path,out_folder:str|Path,batch_size:int=2):
        groups:list[list[Path]] = get_video_groups(
            list(Path(in_folder).glob("*")),
            self.vidlength,
            filename_key=lambda path: os.path.basename(path),
            filename_regex=self.regkey[0],
            format_regex=self.regkey[1],
            idx_key=self.regkey[2],
        )

        dataset_path = Path(f"seqformer_segmentation/vid_dataset_{id(self)}")
        make_video_inference_dataset(groups,dataset_path,images_subfolder_name="images",json_file_path="dataset.json")

        weights_loc = self.pretrain_loc
        if os.path.isdir(weights_loc):
            weights_loc = Path(weights_loc)/"checkpoint.pth"

        predict_custom(str(weights_loc),self.vidlength or len(groups[0]),str(dataset_path/"images"),str(dataset_path/"dataset.json"),"YoutubeVIS")
        

    @classmethod
    def load_model(cls:type["SeqFormerModel"],modelpath:Path|str|os.PathLike[str],modeltype:str,gpu:bool=False,**kwargs)->"LoadableModel":
        assert modeltype == cls.load_type()
        return cls(modelpath,gpu=gpu,**kwargs)

    @classmethod
    def load_type(cls)->str:
        return "SeqFormer"

    @classmethod
    def shortname(cls)->str:
        return "seqformer"

    def delete(self):
        return #nothing in memory when not predicting/training