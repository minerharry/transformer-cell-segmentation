import json
import os
from pathlib import Path
import shutil
import stat
from typing import Callable
import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import BatchFeature
from read_dataset import FOTorchSegmentationDataset, get_meanstd
from torch.utils.data import DataLoader
from transformers.models.maskformer import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
from transformers.models.mask2former import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
import fiftyone as fo

from train_huggingface import HuggingFaceModel
mask2processor = Mask2FormerImageProcessor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)
mask2former_pretrained = {"semantic":"facebook/mask2former-swin-small-ade-semantic","panoptic":"facebook/mask2former-swin-small-cityscapes-panoptic"}


def get_dataloaders(dataset,mean,std,batch_size=2):
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
        batch = mask2processor( #creates a dict with various inputs to the model. Key (get it) arguments: "pixel_values" - raw image input. "mask_labels" - per-class binary masks. "class_labels" - integer label per mask
            images,
            segmentation_maps=segmentation_maps,
            return_tensors="pt",
        )

        for i in range(len(segmentation_maps)):
            if torch.numel(batch["mask_labels"][i]) == 0:
                batch["mask_labels"][i] = torch.zeros((1,) + segmentation_maps[i].shape) #if given a blank mask, produces an empty map
                # from IPython import embed; embed()


        batch["original_images"] = inputs[2]
        batch["original_segmentation_maps"] = inputs[3]
        
        return batch

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_dataloader,test_dataloader




def fine_tune(model:Mask2FormerForUniversalSegmentation|None,dataset:fo.Dataset,mean,std,batch_size=16,epochs=100,save_callback:Callable[[Mask2FormerForUniversalSegmentation],Mask2FormerForUniversalSegmentation]|None=None,autosave_interval=20):
    train_dataloader,test_dataloader = get_dataloaders(dataset,mean,std,batch_size=batch_size)
    
    if model is None:
        model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/maskformer-swin-base-ade",
                                                          id2label=dataset.labels_map,
                                                          ignore_mismatched_sizes=True)

    metric = evaluate.load("mean_iou",nan_to_num=0)

    debug_cpu = False
    device = torch.device("cuda" if not debug_cpu and torch.cuda.is_available() else "cpu")
    print("torch device:",device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    running_loss = 0.0
    num_samples = 0

    for epoch in tqdm(range(epochs)):
        running_loss = 0
        num_samples = 0
        tqdm.write(f"Epoch: {epoch}")
        model.train()
        for idx, batch in enumerate(tqdm(train_dataloader)):
            # Reset the parameter gradients
            optimizer.zero_grad()

            # print("mask:",batch["mask_labels"])
            # print("class:",batch["class_labels"])

            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(device,torch.float),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )

            # Backward propagation
            loss = outputs.loss
            loss.backward()

            batch_size = batch["pixel_values"].size(0)
            running_loss += loss.item()
            num_samples += batch_size

            if idx % 100 == 0:
                tqdm.write(f"Loss: {running_loss/num_samples}")

            # Optimization
            optimizer.step()
        
        # from IPython import embed; embed()
        breakpoint()
        model.eval()
        for idx, batch in enumerate(tqdm(test_dataloader)):

            pixel_values = batch["pixel_values"]
            
            # Forward pass
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values.to(device))

            # get original images
            original_images = batch["original_images"]
            target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
            # predict segmentation maps
            predicted_segmentation_maps = mask2processor.post_process_semantic_segmentation(outputs,
                                                                                        target_sizes=target_sizes)

            # get ground truth segmentation maps
            ground_truth_segmentation_maps = batch["original_segmentation_maps"]

            metric.add_batch(references=ground_truth_segmentation_maps, predictions=predicted_segmentation_maps)
            # from IPython import embed; embed()
            breakpoint()
        
        # NOTE this metric outputs a dict that also includes the mIoU per category as keys
        # so if you're interested, feel free to print them as well
        mIoU = metric.compute(num_labels = 1, ignore_index = 0)['mean_iou']
        # if np.isnan(mIoU):
        #     metric.
        tqdm.write(f"Mean IoU: {mIoU}")

        if epoch % autosave_interval == 0 and save_callback:
            model = save_callback(model)

    return model

def unnormalize_image(pixel_values,mean,std):
    unnormalized_image = (pixel_values * np.array(std)[None, None, None]) + np.array(mean)[None, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    return unnormalized_image

class Mask2FormerModel(HuggingFaceModel):
    @classmethod
    def shortname(cls):
        return "mask2former"
    
    @classmethod
    def load_type(cls):
        return Mask2FormerForUniversalSegmentation;

    def pre_process_semantic_segmentation(self, images, **kwargs) -> BatchFeature:
        return mask2processor(images,**kwargs);

    def post_process_semantic_segmentation(self, outputs, target_sizes: list[tuple] | None = None) -> list:
        return mask2processor.post_process_semantic_segmentation(outputs,target_sizes=target_sizes)
    


# m = Mask2FormerModel();

#just an initializable class to hold state information for evaluation
# class Mask2FormerModel:
#     def __init__(self,modelpath:str|Path,gpu:bool,mean:float|None=None,std:float|None=None):
#         self.model = Mask2FormerForUniversalSegmentation.from_pretrained(modelpath,use_safetensors=False)
#         self.gpu = gpu
#         self.modelpath = Path(modelpath)
#         self._transf = None
#         self._meanstd = (None,None)
#         self.mean = mean
#         self.std = std

#     @property
#     def mean(self):
#         return self.meanstd[0]
    
#     @mean.setter
#     def mean(self,mean:float|None):
#         self.meanstd = (mean,self.meanstd[1])

#     @property
#     def std(self):
#         return self.meanstd[1]
    
#     @std.setter
#     def std(self,std:float|None):
#         self.meanstd = (self.meanstd[0],std)

#     @property
#     def meanstd(self):
#         if self._meanstd[0] is None and self._meanstd[1] is None: #try poll mean,std
#             self._meanstd = self.load_meanstd()
#         return self._meanstd
    
#     @meanstd.setter
#     def meanstd(self,meanstd:tuple[float|None,float|None]):
#         self._meanstd = meanstd
    
#     def load_meanstd(self,meanstdpath:Path|str|None=None):
#         if meanstdpath is None:
#             meanstdpath = Path(self.modelpath).with_name(self.modelpath.stem+"_meanstd.json")
#         meanstdpath = Path(meanstdpath)
#         if meanstdpath.exists():
#             with open(meanstdpath,"r") as f:
#                 mean,std = json.load(f)
#             return (mean,std)
#         else:
#             print(f"Unable to load meanstdpath: {meanstdpath}")
#             return (None,None)
        
#     def save_meanstd(self,meanstdpath:Path|str|None=None):
#         if meanstdpath is None:
#             meanstdpath = Path(self.modelpath).with_name(self.modelpath.stem+"_meanstd.json")
#         meanstdpath = Path(meanstdpath)
#         meanstdpath.parent.mkdir(parents=True,exist_ok=True);
#         with open(meanstdpath,"w") as f:
#             json.dump([self.mean,self.std],f)


#     @property
#     def transf(self):
#         if self._transf is None and self.mean is not None and self.std is not None:
#             import albumentations as A
#             self._transf = A.Normalize(self.mean,self.std)
#         return self._transf
        
    
#     #make it satisfy LoadableModel protocol
#     @classmethod
#     def load_model(cls,modelpath:Path|str|os.PathLike[str],modeltype:str,gpu:bool=False,**kwargs)->"Mask2FormerModel": #I want python 3.11 so I can use Self...
#         assert modeltype == "mask2former"
#         return Mask2FormerModel(modelpath,gpu)
        
#     @classmethod
#     def load_type(cls):
#         return "mask2former"
    
#     def fine_tune(self,dataset:fo.Dataset,batch_size:int=2,epochs:int=100,**kwargs)->"Mask2FormerModel":
#         if self.mean is None or self.std is None:
#             self.update_meanstd(dataset)

#         def save(model):
#             self.model = model
#             try:
#                 self.save_model()
#             except:
#                 pass
#             return self.model

#         if not kwargs.get("do_autosave",False):
#             save = None

#         self.model = fine_tune(self.model,dataset,self.mean,self.std,batch_size=batch_size,epochs=epochs,save_callback=save);
#         return self
    
#     def update_meanstd(self,dataset:fo.Dataset):
#         self.mean,self.std = get_meanstd(dataset)

#     #make it satisfy cell_segmentation predict protocol.
#     def predict(self,x:np.ndarray,verbose:bool=True)->torch.Tensor:
#         self.model.eval();
#         if self.transf is None:
#             raise ValueError("Mean and standard deviation of dataset must be provided!")
#         im = self.transf(image=x)["image"];
        
#         batch = mask2processor(list(im),data_former="CHANNELS_LAST")
        

#         if self.gpu:
#             self.model.to("cuda")

#         with torch.no_grad():
#             outputs = self.model(torch.tensor(batch["pixel_values"]).to("cuda" if self.gpu else "cpu"))

#         output_mask = torch.stack(mask2processor.post_process_semantic_segmentation(outputs,target_sizes=[im.shape[:2] for im in x]))

#         # from IPython import embed; embed()

#         print(output_mask.shape)
#         print(torch.count_nonzero(output_mask)/output_mask.numel())

#         return output_mask.to("cpu");

#     def save_model(self,savepath:str|Path|None=None):
#         save = savepath or self.modelpath
#         device = next(self.model.parameters()).device
#         self.model.to("cpu")
        
#         # from IPython import embed; embed()
        
#         self.model.save_pretrained("dummy_stupid_temp_model_folder",safe_serialization=False)
#         del self.model
#         rmdir(save)
#         self.model = Mask2FormerForUniversalSegmentation.from_pretrained("dummy_stupid_temp_model_folder")
#         self.model.save_pretrained(save,safe_serialization=False)
#         rmdir("dummy_stupid_temp_model_folder")
#         self.modelpath = Path(save)
#         self.model.to(device)
#         print("model saved")

#         if self.mean is not None and self.std is not None:
            # self.save_meanstd()

#brute force    
def on_rm_error( func, path, exc_info):
    # path contains the path of the file that couldn't be removed
    # let's just assume that it's read-only and unlink it.
    os.chmod( path, stat.S_IWRITE )

def rmdir(dir):
    shutil.rmtree(dir,onerror=on_rm_error)