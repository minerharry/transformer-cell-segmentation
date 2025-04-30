##Use segmentation stuff from cell_segmentation.py to preprocess (for stitching and 3-stacking) data for the various models
from enum import Enum
import functools
import os
import sys

from read_dataset import get_dataset
from train_mask2former import Mask2FormerModel
cell_segmentation_path = r"C:\Users\miner\Documents\Github\cell-segmentation"
sys.path.insert(0,cell_segmentation_path)

from imageio.v3 import imwrite
import fiftyone as fo

from pathlib import Path
from typing import Callable, Literal, Protocol
from cell_segmentation_training import proc_type
from segment_movie import Model, SourceProcessor, segment_images, get_segmentation_sequence, SegmentationParams, make_process_fn, make_deprocess_fn

MODELS_FOLDER = "gs://optotaxisbucket/models/transformers"
GCP_FOLDER = "gcp_transfer"

class LoadableModel(Model,Protocol):
    def fine_tune(self,dataset:fo.Dataset,batch_size:int=2,epochs:int=100,**kwargs)->"LoadableModel": ...

    def save_model(self,savepath:str|Path|None=None): ...

    @classmethod
    def load_model(cls,modelpath:Path|str|os.PathLike[str],modeltype:str,gpu:bool=False,**kwargs)->"LoadableModel": ... #I want python 3.11 so I can use Self...

    @classmethod
    def load_type(cls)->str:...



def make_training_dataset(data_folder:Path|str,dataset_key,im_proc:proc_type,mask_proc:proc_type):
    images = Path(data_folder)/"images"
    masks = Path(data_folder)/"masks"
    source = SourceProcessor(
                training_images=images,
                training_masks=masks,
                gcp_transfer=GCP_FOLDER,
                local=f"training/{dataset_key}",
                clear_on_exit=False)
    source.create()

    
    
    processed_images = source.process_training_images(im_proc)
    processed_masks = source.process_training_masks(mask_proc)

    return processed_images,processed_masks




def segment_data(model:type[LoadableModel],modelname,input_images,output_masks,proc_fn,deproc_fn,clear_on_close:bool=True,model_kwargs:dict={}): #GOD python is good, Protocol subtypes also count for protocol subclasses. fuck yeah
    segment_images(modelname,
                   input_images,output_masks,
                   proc_fn,deproc_fn,
                   modeltype=model.load_type(),
                   load_model=functools.partial(model.load_model,**model_kwargs),
                   models_folder=MODELS_FOLDER,gcp_transfer=GCP_FOLDER,local_folder="local",
                   clear_on_close=clear_on_close)
    


process_presets = {
    "optotaxis":{
        "3stack_nosplit":  SegmentationParams(stack_images=True,do_splitting=False),
        "3stack_split":    SegmentationParams(stack_images=True,do_splitting=True),
        "nostack_split":   SegmentationParams(stack_images=False,do_splitting=True),
        "nostack_nosplit": SegmentationParams(stack_images=False,do_splitting=False),
    }
}


def fine_tune(dataset_name:str,dataset_setting:str,
              model_type:type[LoadableModel],modelpath:Path|str,in_modelsfolder:bool=True,
              batch_size:int=16,epochs:int=100,gpu:bool=True, save:str|Path|bool=False, save_modelsfolder:bool=True,
              do_autosave:bool=True,
              ):
    data_folder = Path("datasets")/dataset_name
    dataparams = process_presets[dataset_name][dataset_setting]

    im_proc = make_process_fn(dataparams,"images")
    mask_proc = make_process_fn(dataparams,"masks")

    imfolder,maskfolder = make_training_dataset(data_folder,f"{dataset_name}/{dataset_setting}",im_proc,mask_proc);

    dataset = get_dataset(imfolder,maskfolder)

    dataset.info["data_folder"] = str(data_folder)
    dataset.save()

    if in_modelsfolder or save_modelsfolder:
        source = SourceProcessor(models=MODELS_FOLDER,gcp_transfer=GCP_FOLDER,local=f"local"); #modelsfolder goes toplevel
        source.create() #since we're not doing it as a context manager!
        modelsfolder = source.fetch_modelsfolder()
        if in_modelsfolder:
            modelpath = modelsfolder/modelpath
    else:
        source = None
        modelsfolder = None

    model = model_type.load_model(modelpath,model_type.load_type(),gpu=gpu)

    if save and do_autosave: #initial autosave; sets save path for autosaving
        try:
            Path(save)
        except:
            #save just a bool, use original save location
            model.save_model()
        else:
            if save_modelsfolder:
                save = Path(save)
                assert modelsfolder is not None
                assert source is not None
                save = modelsfolder/save
                model.save_model(save)
            else:
                model.save_model(save)

    try:
        model.fine_tune(dataset,batch_size=batch_size,epochs=epochs,do_autosave=do_autosave);
    finally:
        if save:
            try:
                Path(save)
            except:
                #save just a bool, use original save location
                model.save_model()
            else:
                if save_modelsfolder:
                    save = Path(save)
                    assert modelsfolder is not None
                    assert source is not None
                    save = modelsfolder/save
                    model.save_model(save)
                    source.push_modelsfolder()
                else:
                    model.save_model(save)
    
    return model

if __name__ == "__main__":
    from train_mask2former import mask2former_pretrained

    # fine_tune("optotaxis","nostack_split",Mask2FormerModel,mask2former_pretrained["semantic"],in_modelsfolder=False,save_modelsfolder=True,save="mask2former_nostack_split_optotaxis_2",
    #           epochs=1); #just to download for autosave purposes

    # fine_tune("optotaxis","3stack_split",Mask2FormerModel,mask2former_pretrained["semantic"],in_modelsfolder=False,save_modelsfolder=True,save="mask2former_3stack_split_optotaxis_2",
    #           epochs=1); #just to download for autosave purposes


    fine_tune("optotaxis","nostack_split",Mask2FormerModel,"mask2former_nostack_split_optotaxis_2",in_modelsfolder=True,save_modelsfolder=True,save="mask2former_nostack_split_optotaxis_2",
              epochs=100);


    fine_tune("optotaxis","3stack_split",Mask2FormerModel,"mask2former_3stack_split_optotaxis_2",in_modelsfolder=True,save_modelsfolder=True,save="mask2former_3stack_split_optotaxis_2",
              epochs=100);