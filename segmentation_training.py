##Use segmentation stuff from cell_segmentation.py to preprocess (for stitching and 3-stacking) data for the various models
from enum import Enum
import functools
import os
import sys

from tqdm import tqdm

from other_models import OneFormerModel, SegFormerModel
from read_dataset import get_dataset
from train_mask2former import Mask2FormerModel
cell_segmentation_path = r"C:\Users\miner\Documents\Github\cell-segmentation"
sys.path.insert(0,cell_segmentation_path)

from imageio.v3 import imwrite
import fiftyone as fo

from pathlib import Path, PurePath
from typing import Any, Callable, Generic, Literal, Protocol, TypeVar
from cell_segmentation_training import proc_type,is_gcp_path
from segment_movie import Model, SourceProcessor, segment_images, get_segmentation_sequence, SegmentationParams, make_process_fn, make_deprocess_fn, ModelLoader, load_model

MODELS_FOLDER = "gs://optotaxisbucket/models/transformers"
GCP_FOLDER = "gcp_transfer"

T=TypeVar("T")
class LoadableModel(Model,Generic[T],Protocol):
    def fine_tune(self,dataset:fo.Dataset,batch_size:int=2,epochs:int=100,**kwargs)->"LoadableModel": ...

    def save_model(self,savepath:str|Path|None=None): ...

    @classmethod
    def load_model(cls,modelpath:Path|str|os.PathLike[str],modeltype:T,gpu:bool=False,**kwargs)->"LoadableModel": ... #I want python 3.11 so I can use Self...

    @classmethod
    def load_type(cls)->T:...

    @classmethod
    def shortname(cls)->str: ...

    def delete(self): ...



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


def segment_dataset(modelname:str,
                   im_source:str|Path,
                   mask_out:str|Path|tuple[str|Path,...], 
                   proc_fn:proc_type,
                   deproc_fn:proc_type,
                   batch_size:int=32,
                   modeltype:str="keras",
                   load_model:ModelLoader=load_model,
                   models_folder:str|Path="gs://optotaxisbucket/models",
                   local_folder:str|Path|None=None,
                   gcp_transfer:str|Path|None=None,
                   clear_on_close:bool=False,
                   gpu=True):

    with SourceProcessor(models=models_folder,
                             segmentation_images=im_source,
                             local=local_folder,
                             gcp_transfer=gcp_transfer,
                             segmentation_masks=mask_out,
                             clear_on_exit=clear_on_close) as source:

        print("processing images...")
        im_folder = source.process_segmentation_images(proc_fn)
        out_folder = Path(source.get_local("segmentation_masks"))
        print("image processing complete")
        
        # print("segmenting to output folder:",out_folder)
        # seg_sequence = get_segmentation_sequence(im_folder,out_folder,batch_size=batch_size)
        

        modelpath = source.fetch_modelsfolder()
        model = load_model(Path(modelpath)/modelname,modeltype,gpu=gpu,compile_args=dict(optimizer="rmsprop",loss=None,))

        model.predict_dataset(im_folder,out_folder,batch_size=batch_size)

        # print(f"predicting {len(seg_sequence)*batch_size} masks...")    
        # for batch,inpaths in tqdm(seg_sequence,desc="predicting", dynamic_ncols=True):
        #     predictions:torch.Tensor = model.predict(batch,verbose=(len(batch) <= 1))
        #     print("preds_shape:",predictions.shape)
        #     for path,prediction in zip(inpaths,predictions):
        #         name = Path(path).relative_to(im_folder)
        #         im = prediction.numpy().astype('uint8');
        #         imwrite(out_folder/name,im);
        print("prediction complete")

        print("deprocessing masks...")
        source.deprocess_segmentation_masks(deproc_fn)
    print("mask deprocessing complete. Segmentation complete")


def segment_data(model:type[LoadableModel],modelname,input_images,output_masks,proc_fn,deproc_fn,clear_on_close:bool=True,model_kwargs:dict={},as_dataset=False): #GOD python is good, Protocol subtypes also count for protocol subclasses. fuck yeah
    if as_dataset:
        segment = segment_dataset
    else:
        segment = segment_images
    segment(modelname,
                   input_images,output_masks,
                   proc_fn,deproc_fn,
                   modeltype=model.load_type(),
                   load_model=functools.partial(model.load_model,**model_kwargs),
                   models_folder=MODELS_FOLDER,gcp_transfer=GCP_FOLDER,local_folder="local",
                   clear_on_close=clear_on_close)



def fine_tune(dataset_name:str,dataset_setting:str,
              model_type:type[LoadableModel],modelpath:Path|str,in_modelsfolder:bool=True,
              batch_size:int=16,epochs:int=100,gpu:bool=True, save:str|Path|bool=False, save_modelsfolder:bool=True,
              do_autosave:bool=True, autosave_interval:int=20,
              **load_kwargs,
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

    model = model_type.load_model(modelpath,model_type.load_type(),gpu=gpu,**load_kwargs)

    if save and do_autosave: #initial autosave; sets save path for autosaving
        try:
            Path(save)
        except:
            #save just a bool, use original save location
            model.save_model()
        else:
            if save_modelsfolder or in_modelsfolder:
                save = Path(save)
                assert modelsfolder is not None
                assert source is not None
                save = modelsfolder/save
                model.save_model(save)
            else:
                model.save_model(save)

    try:
        model.fine_tune(dataset,batch_size=batch_size,epochs=epochs,do_autosave=do_autosave,autosave_interval=autosave_interval);
    finally:
        if sys.exc_info()[0] is not None:
            import traceback as tb
            ex = tb.format_exc()
            tqdm.write(ex)
        if save:
            try:
                Path(save)
            except:
                #save just a bool, use original save location
                model.save_model()
            else:
                if save_modelsfolder or in_modelsfolder: 
                    save = Path(save)
                    assert modelsfolder is not None
                    save = modelsfolder/save
                    model.save_model(save)
                else:
                    model.save_model(save)
            if save_modelsfolder or in_modelsfolder:
                    assert source is not None
                    source.push_modelsfolder()
    

    print("Fine tuning completed successfully")
    return model


def train_model(model:type[LoadableModel],dataset:str,setting:str,suffix:str="",remote_model:str|None=None,force_remote:bool=False,modelsfolder:str=MODELS_FOLDER,save=True,**kwargs):
    modelname = f"{model.shortname()}_{setting}_{dataset}{suffix}"
    # breakpoint()
    if is_gcp_path(PurePath(modelsfolder)):
        in_modelsfolder=True
        source = SourceProcessor(models=modelsfolder,gcp_transfer=GCP_FOLDER,local=f"local",clear_on_exit=False);
        source.create()
        folder = source.fetch_modelsfolder()
        local_modelpath = Path(folder)/modelname
        modelpath = modelname
    else:
        source = None
        in_modelsfolder=False
        local_modelpath = Path(modelsfolder)/modelname
        modelpath = str(local_modelpath)

    # from IPython import embed; embed()
    print(local_modelpath)
    if not os.path.exists(local_modelpath) or force_remote: #need to download from remote
        print(f"fetching remote pretrained model: {remote_model}")
        assert remote_model is not None
        mod = model.load_model(remote_model,model.load_type(),**kwargs)
        mod.save_model(local_modelpath)
        mod.delete()

    # from IPython import embed; embed()
    
    if in_modelsfolder:
        assert source is not None
        source.push_modelsfolder()

    return fine_tune(dataset,setting,model,modelpath,in_modelsfolder=in_modelsfolder,save_modelsfolder=in_modelsfolder,save=save,**kwargs)








from libraries.filenames import filename_regex_anybasename, filename_regex_format
optotaxis_regkey = (
    filename_regex_anybasename,
    filename_regex_format,
    3
)

stardist_10x_regkey = ( #parse_regex, format_regex, idx_key
    r"(.*)(_|-)(\d*)\.tif$",
    r"{}{}{}\.tif$",
    2
)

stardist_20x_regkey = (
    r"([a-z])(\d*)\.tif$",
    r"{}{}\.tif$",
    1
)

def get_enumerator_regkey(regkey:tuple[str,str,int]): #enumerator adds "-{#}.{ext}" to end. parse this precisely
    is_fullmatch = regkey[0].endswith("$") or regkey[1].endswith("$") #really they should always be in agreement but

    suffix_regex = r"-(\d+)\.(.+)" + ("$" if is_fullmatch else "")
    suffix_format = r"-{}\.{}" + ("$" if is_fullmatch else "")

    return (regkey[0].rstrip("$") + suffix_regex, regkey[1].rstrip("$") + suffix_format, regkey[2]) #since groups added to the end, don't change index location

optotaxis_enum_regkey = get_enumerator_regkey(optotaxis_regkey)
stardist_10x_enum_regkey = get_enumerator_regkey(stardist_10x_regkey)
stardist_20x_enum_regkey = get_enumerator_regkey(stardist_20x_regkey)


process_presets = {
    "optotaxis":{
        "3stack_nosplit":  SegmentationParams(stack_images=True,do_splitting=False),
        "3stack_split":    SegmentationParams(stack_images=True,do_splitting=True),
        "nostack_split":   SegmentationParams(stack_images=False,do_splitting=True),
        "nostack_nosplit": SegmentationParams(stack_images=False,do_splitting=False),
    },
        "optotaxis_half":{
        "3stack_nosplit":  SegmentationParams(stack_images=True,do_splitting=False),
        "3stack_split":    SegmentationParams(stack_images=True,do_splitting=True),
        "nostack_split":   SegmentationParams(stack_images=False,do_splitting=True),
        "nostack_nosplit": SegmentationParams(stack_images=False,do_splitting=False),
    },
        "optotaxis_quarter":{
        "3stack_nosplit":  SegmentationParams(stack_images=True,do_splitting=False),
        "3stack_split":    SegmentationParams(stack_images=True,do_splitting=True),
        "nostack_split":   SegmentationParams(stack_images=False,do_splitting=True),
        "nostack_nosplit": SegmentationParams(stack_images=False,do_splitting=False),
    },
    "stardist_10x":{ #for now, only semantic
        "3stack_split": SegmentationParams(stack_images=True,do_splitting=True,x_slices=5,y_slices=5,dx=32,dy=31,stacking_regex_key=stardist_10x_regkey),
        "nostack_split": SegmentationParams(stack_images=False,do_splitting=True,x_slices=5,y_slices=5,dx=32,dy=31,stacking_regex_key=stardist_10x_regkey),
    },
    "stardist_20x":{ #for now, only semantic
        "3stack_split": SegmentationParams(stack_images=True,do_splitting=True,x_slices=5,y_slices=5,dx=32,dy=31,stacking_regex_key=stardist_20x_regkey),
        "nostack_split": SegmentationParams(stack_images=False,do_splitting=True,x_slices=5,y_slices=5,dx=32,dy=31,stacking_regex_key=stardist_20x_regkey),
    },
    "stardist_joint":{ #for now, only semantic
        "3stack_split": SegmentationParams(stack_images=True,do_splitting=True,x_slices=5,y_slices=5,dx=32,dy=31,stacking_regex_key=stardist_20x_regkey),
        "nostack_split": SegmentationParams(stack_images=False,do_splitting=True,x_slices=5,y_slices=5,dx=32,dy=31,stacking_regex_key=stardist_20x_regkey),
    }
}


if __name__ == "__main__":
    from train_mask2former import mask2former_pretrained

    # fine_tune("optotaxis","nostack_split",Mask2FormerModel,mask2former_pretrained["semantic"],in_modelsfolder=False,save_modelsfolder=True,save="mask2former_nostack_split_optotaxis_2",
    #           epochs=1); #just to download for autosave purposes

    # fine_tune("optotaxis","3stack_split",Mask2FormerModel,mask2former_pretrained["semantic"],in_modelsfolder=False,save_modelsfolder=True,save="mask2former_3stack_split_optotaxis_2",
    #           epochs=1); #just to download for autosave purposes


    # fine_tune("optotaxis","nostack_split",Mask2FormerModel,"mask2former_nostack_split_optotaxis_2",in_modelsfolder=True,save_modelsfolder=True,save="mask2former_nostack_split_optotaxis_2",
    #           epochs=2,batch_size=4);


    # fine_tune("optotaxis","3stack_split",Mask2FormerModel,"mask2former_3stack_split_optotaxis_2",in_modelsfolder=True,save_modelsfolder=True,save="mask2former_3stack_split_optotaxis_2",
    #           epochs=100);
    

    # fine_tune("optotaxis","3stack_split",SegFormerModel,"segformer_3stack_split_optotaxis",in_modelsfolder=False,save_modelsfolder=True,save="segformer_3stack_split_optotaxis",
    #           epochs=100,batch_size=4);

    # fine_tune("optotaxis","nostack_split",SegFormerModel,"segformer_nostack_split_optotaxis",in_modelsfolder=True,save_modelsfolder=True,save="segformer_nostack_split_optotaxis",
    #           epochs=100,batch_size=4);


    # train_model(SegFormerModel,"stardist_10x","3stack_split",epochs=100,batch_size=8,autosave_interval=5)

    ### mass training

    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("segmentation_training.log"),
                            logging.StreamHandler()
                        ])

    enum_regkeys = {
        "stardist_20x":stardist_20x_enum_regkey,
        "stardist_10x":stardist_10x_enum_regkey,
        "optotaxis":optotaxis_enum_regkey
    }

    from seqformer import SeqFormerModel
    models = [(SegFormerModel,"nvidia/mit-b0")]
    # models = [(SeqFormerModel,r"C:\Users\miner\Documents\Github\SeqFormer\pretrained\swinL_weight.pth")]
    datasets = ["stardist_joint"]#"stardist_20x","stardist_10x","optotaxis_quarter","optotaxis_half","optotaxis"]#
    settings = ["3stack_split","nostack_split"]
    
    epochs = 50
    bs = 4
    save_interval = 10

    startcycle = 0

    cycle_idx = 0
    for (modtype,modremote) in models:
        for dataset in datasets:
            for setting in settings:
                cycle_idx += 1
                if cycle_idx <= startcycle: #in case it breaks partway through
                    continue
                train_model(modtype,dataset,setting,epochs=20,batch_size=4,autosave_interval=5,remote_model=modremote)#,regkey=enum_regkeys[dataset])
                with open("cycle_log.txt","+a") as f:
                    f.write(f"Cycle {cycle_idx} completed\n")