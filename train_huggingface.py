from abc import ABC, abstractmethod
from datetime import datetime
import operator
from pathlib import Path
import shutil
import stat
from typing import Any, Callable, Generic, Protocol, TypeVar
import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.image_processing_utils import BaseImageProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils_fast import SemanticSegmentationMixin
import fiftyone as fo
import json
import os

from read_dataset import FOTorchSegmentationDataset, get_meanstd

#just an initializable class to hold state information for evaluation
class HuggingFaceModel(ABC):
    def __init__(self,Model:type[PreTrainedModel],modelpath:str|Path,gpu:bool,mean:float|None=None,std:float|None=None,**kwargs):
        self.modeltype = Model;
        self.model = self.load_pretrained(Model,modelpath,gpu,**kwargs)
        self.gpu = gpu
        self.modelpath = Path(modelpath)
        self._transf = None
        self._meanstd = (None,None)
        self.mean = mean
        self.std = std

    def load_pretrained(self,Model:type[PreTrainedModel],modelpath:str|Path,gpu:bool,**kwargs):
        return Model.from_pretrained(modelpath,use_safetensors=False,trust_remote_code=True)

    @property
    def mean(self):
        return self.meanstd[0]
    
    @mean.setter
    def mean(self,mean:float|None):
        self.meanstd = (mean,self.meanstd[1])

    @property
    def std(self):
        return self.meanstd[1]
    
    @std.setter
    def std(self,std:float|None):
        self.meanstd = (self.meanstd[0],std)

    @property
    def meanstd(self):
        if self._meanstd[0] is None and self._meanstd[1] is None: #try poll mean,std
            self._meanstd = self.load_meanstd()
        return self._meanstd
    
    @meanstd.setter
    def meanstd(self,meanstd:tuple[float|None,float|None]):
        self._meanstd = meanstd
    
    def load_meanstd(self,meanstdpath:Path|str|None=None):
        if meanstdpath is None:
            meanstdpath = Path(self.modelpath).with_name(self.modelpath.stem+"_meanstd.json")
        meanstdpath = Path(meanstdpath)
        if meanstdpath.exists():
            with open(meanstdpath,"r") as f:
                mean,std = json.load(f)
            return (mean,std)
        else:
            print(f"Unable to load meanstdpath: {meanstdpath}")
            return (None,None)
        
    def save_meanstd(self,meanstdpath:Path|str|None=None):
        if meanstdpath is None:
            meanstdpath = Path(self.modelpath).with_name(self.modelpath.stem+"_meanstd.json")
        meanstdpath = Path(meanstdpath)
        meanstdpath.parent.mkdir(parents=True,exist_ok=True);
        with open(meanstdpath,"w") as f:
            json.dump([self.mean,self.std],f)


    @property
    def transf(self):
        if self._transf is None and self.mean is not None and self.std is not None:
            import albumentations as A
            self._transf = A.Normalize(self.mean,self.std)
        return self._transf
        
    
    #make it satisfy LoadableModel protocol
    @classmethod
    def load_model(cls,modelpath:Path|str|os.PathLike[str],modeltype:type[PreTrainedModel],gpu:bool=False,**kwargs)->"HuggingFaceModel": #I want python 3.11 so I can use Self...
        assert modeltype == cls.load_type();
        return cls(modeltype,modelpath,gpu)
        
    @classmethod
    @abstractmethod
    def load_type(cls)->type[PreTrainedModel]: ...
    
    def fine_tune(self,dataset:fo.Dataset,batch_size:int=2,epochs:int=100,**kwargs)->"HuggingFaceModel":
        do_autosave = kwargs.pop("do_autosave",True)
        if self.mean is None or self.std is None:
            self.update_meanstd(dataset)

        def save(model):
            self.model = model
            try:
                self.save_model()
            except:
                pass
            return self.model

        if not kwargs.get("do_autosave",False):
            save = None

        self.model = hf_fine_tune_semantic(self.model,dataset,self.mean,self.std,
                                           self.pre_process_semantic_segmentation,self.post_process_semantic_segmentation,
                                           batch_size=batch_size,epochs=epochs,save_callback=save if do_autosave else None,**kwargs);
        return self
    
    def update_meanstd(self,dataset:fo.Dataset):
        self.mean,self.std = get_meanstd(dataset)

    #make it satisfy cell_segmentation predict protocol.
    def predict(self,x:np.ndarray,verbose:bool=True)->torch.Tensor:
        # breakpoint()
        assert self.model is not None
        self.model.eval();
        if self.transf is None:
            raise ValueError("Mean and standard deviation of dataset must be provided!")
        im = self.transf(image=x)["image"];
        
        batch = self.pre_process_semantic_segmentation(list(im),data_former="CHANNELS_LAST")
        
        if self.gpu:
            self.model.to("cuda")

        with torch.no_grad():
            outputs = self.model(torch.as_tensor(np.array(batch["pixel_values"]),device="cuda" if self.gpu else "cpu"))#.to("cuda" if self.gpu else "cpu"))

        output_mask = torch.stack(self.post_process_semantic_segmentation(outputs,target_sizes=[im.shape[:2] for im in x]))

        # from IPython import embed; embed()

        print(output_mask.shape)
        print(torch.count_nonzero(output_mask)/output_mask.numel())

        return output_mask.to("cpu");

    def save_model(self,savepath:str|Path|None=None):
        # breakpoint()
        assert self.model is not None
        save = savepath or self.modelpath
        device = next(self.model.parameters()).device
        self.model.to("cpu")
        
        # from IPython import embed; embed()
        
        dummy_path = f"dummy_stupid_temp_model_folder_{id(self)}"
        self.model.save_pretrained(dummy_path,safe_serialization=False)
        del self.model
        if os.path.exists(save):
            rmdir(save)
        self.model = self.load_pretrained(self.modeltype,dummy_path,False)
        self.model.save_pretrained(save,safe_serialization=False)
        rmdir(dummy_path)
        self.modelpath = Path(save)
        self.model.to(device)
        print("model saved")

        with open(self.modelpath.with_suffix(".modified.txt"),"a") as f:
            f.write(f"{datetime.now()}\n")

        if self.mean is not None and self.std is not None:
            self.save_meanstd()
        

    def to(self,*args):
        self.model.to(*args)
        return self

    @abstractmethod
    def pre_process_semantic_segmentation(self,images,**kwargs)->BatchFeature: ...

    @abstractmethod
    def post_process_semantic_segmentation(self, outputs, target_sizes: list[tuple]|None = None)->list: ...

    @abstractmethod
    def shortname(cls)->str:...

    def delete(self):
        del self.model



#brute force    
def on_rm_error( func, path, exc_info):
    # path contains the path of the file that couldn't be removed
    # let's just assume that it's read-only and unlink it.
    os.chmod( path, stat.S_IWRITE )

def rmdir(dir):
    shutil.rmtree(dir,onerror=on_rm_error)




def get_dataloaders_hf(dataset,mean,std,preprocessor,device:str|torch.device=torch.device("cpu"),batch_size=2):
    import albumentations as A
    
    transform = A.Normalize(mean=mean, std=std)
    train_dataset = FOTorchSegmentationDataset(dataset.match_tags("train"),classes="cytoplasm",transforms=transform)
    test_dataset = FOTorchSegmentationDataset(dataset.match_tags("val"),classes="cytoplasm",transforms=transform)

    # from IPython import embed; embed()


    def collate_fn(batch):
        inputs = list(zip(*batch))
        images = inputs[0]
        segmentation_maps = inputs[1]
        # this function pads the inputs to the same size,
        # and creates a pixel mask
        # actually padding isn't required here since we are cropping
        # print(np.array(images).shape)
        batch = preprocessor( #creates a dict with various inputs to the model. Key (get it) arguments: "pixel_values" - raw image input. "mask_labels" - per-class binary masks. "class_labels" - integer label per mask
            images,
            segmentation_maps=segmentation_maps,
            return_tensors="pt",
        ).to(device)

        for i in range(len(segmentation_maps)):
            if "mask_labels" in batch:
                if torch.numel(batch["mask_labels"][i]) == 0:
                    batch["mask_labels"][i] = torch.zeros((1,) + segmentation_maps[i].shape) #if given a blank mask, produces an empty map
                    # from IPython import embed; embed()


        batch["original_images"] = inputs[2]
        batch["original_segmentation_maps"] = inputs[3]
        
        return batch

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_dataloader,test_dataloader

M = TypeVar("M",bound=PreTrainedModel)
from torch.nn.modules.loss import _Loss
def hf_fine_tune_semantic(model:M,dataset:fo.Dataset,mean,std,preprocessor,postprocessor,batch_size=16,epochs=100,save_callback:Callable[[M],M]|None=None,loss_fn:Callable[[Any],_Loss]=operator.attrgetter("loss"),autosave_interval=20):
    print("autosave interval:",autosave_interval)
    
    # if model is None:
    #     model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/maskformer-swin-base-ade",
    #                                                       id2label=dataset.labels_map,
    #                                                       ignore_mismatched_sizes=True)

    metric = evaluate.load("mean_iou",nan_to_num=0)

    debug_cpu = False
    device = torch.device("cuda" if not debug_cpu and torch.cuda.is_available() else "cpu")
    print("torch device:",device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) #default was 5e-5 fyi

    running_loss = 0.0
    num_samples = 0

    train_dataloader,test_dataloader = get_dataloaders_hf(dataset,mean,std,preprocessor,device=device,batch_size=batch_size)

    for epoch in tqdm(range(epochs)):
        running_loss = 0
        num_samples = 0
        tqdm.write(f"Epoch: {epoch}")
        model.train()
        for idx, batch in enumerate(tqdm(train_dataloader)):
            # Reset the parameter gradients
            optimizer.zero_grad()
            # breakpoint()
            # from IPython import embed; embed()

            # print("mask:",batch["mask_labels"])
            # print("class:",batch["class_labels"])

            # Forward pass
            # if isinstance(batch,dict):
            d = dict(**batch)
            del d["original_images"]
            del d["original_segmentation_maps"]
            # breakpoint()
            # from IPython import embed; embed()
            outputs = model(**d)
            # else:
            #     outputs = model()

            # Backward propagation
            loss = loss_fn(outputs)
            loss.backward()

            running_loss += loss.item()
            num_samples += batch_size

            if idx % 100 == 0:
                tqdm.write(f"Loss: {running_loss/num_samples}")

            # Optimization
            optimizer.step()
        
        # from IPython import embed; embed()
        # breakpoint()
        # model.eval()
        # for idx, batch in enumerate(tqdm(test_dataloader)):

        #     pixel_values = batch["pixel_values"]
            
        #     # Forward pass
        #     with torch.no_grad():
        #         outputs = model(pixel_values=pixel_values.to(device))

        #     # get original images
        #     original_images = batch["original_images"]
        #     target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
        #     # predict segmentation maps
        #     predicted_segmentation_maps = postprocessor(outputs,target_sizes=target_sizes)

        #     # get ground truth segmentation maps
        #     ground_truth_segmentation_maps = batch["original_segmentation_maps"]

        #     metric.add_batch(references=ground_truth_segmentation_maps, predictions=predicted_segmentation_maps)
        #     # from IPython import embed; embed()
        #     # breakpoint()
        
        # # NOTE this metric outputs a dict that also includes the mIoU per category as keys
        # # so if you're interested, feel free to print them as well
        # mIoU = metric.compute(num_labels = 1, ignore_index = 0)['mean_iou']
        # # if np.isnan(mIoU):
        # #     metric.
        # tqdm.write(f"Mean IoU: {mIoU}")

        if epoch % autosave_interval == 0 and save_callback:
            model = save_callback(model)

    return model