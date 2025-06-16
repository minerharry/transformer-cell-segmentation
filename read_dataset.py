
import json
from typing import Iterable, Optional, Tuple#, TypeVarTuple
import fiftyone as fo
from fiftyone.utils.data.importers import ImageSegmentationDirectoryImporter
from pathlib import Path
import fiftyone.utils.random as four
from imageio.v3 import imread
import numpy as np
import torch
from skimage.exposure import rescale_intensity
import fiftyone.core.metadata as fom
import fiftyone.core.labels as fol


def shared_files(*paths:Iterable[str|Path]) -> list[Path]:
    return list(set.intersection(*(set(map(Path,ps)) for ps in paths)))


class FOTorchSegmentationDataset(torch.utils.data.Dataset):
    """A class to construct a PyTorch dataset from a FiftyOne dataset containing segmantic segmentation masks.
    
    Args:
        fiftyone_dataset: a FiftyOne dataset or view that will be used for 
            training or testing
        transforms (None): a list of PyTorch transforms to apply to images 
            and targets when loading
        gt_field ("ground_truth"): the name of the field in fiftyone_dataset 
            that contains the desired labels to load
        classes (None): a list of class strings that are used to define the 
            mapping between class names and indices. If None, it will use 
            all classes present in the given fiftyone_dataset.
    """

    def __init__(
        self,
        fiftyone_dataset:fo.Dataset,
        classes:list[str]|str,
        transforms=None,
        gt_field="ground_truth",
    ):
        self.samples = fiftyone_dataset
        self.transforms = transforms
        self.gt_field = gt_field

        self.img_paths = self.samples.values("filepath")
        self.mask_paths = self.samples.values(f"{gt_field}.mask_path")
        self.classes = [classes] if isinstance(classes,str) else classes

        if self.classes[0] != "background":
            self.classes = ["background"] + self.classes

        self.labels_map = {i: c for i, c in enumerate(self.classes)}
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = imread(img_path)

        mask_path = self.mask_paths[idx]
        mask = imread(mask_path)
        mask = mask > 0 #reads mask 0,255 otherwise


        if self.transforms is not None:
            t = self.transforms(image=img, mask=mask)
            timg = t["image"]
            tmask = t["mask"]

        # from IPython import embed; embed()

        return timg, tmask, img, mask

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes


##importer which requires both masks __and__ images to be present to import. 
class SharedImageMaskImporter(ImageSegmentationDirectoryImporter):
    def __len__(self):
        raise ValueError("__len__ not properly defined sorry")
    
    def __next__(self):
        uuid = None
        while uuid not in self._image_paths_map: #find next uuid
            uuid = next(self._iter_uuids)

        image_path = self._image_paths_map[uuid]
        mask_path = self._labels_paths_map.get(uuid, None)

        if self.compute_metadata:
            image_metadata = fom.ImageMetadata.build_for(image_path)
        else:
            image_metadata = None

        if mask_path is not None:
            label = fol.Segmentation(mask_path=mask_path)
            if self.load_masks:
                label.import_mask(update=True)
                if self.force_grayscale and label.mask.ndim > 1:
                    label.mask = label.mask[:, :, 0]
        else:
            label = None

        return image_path, image_metadata, label


def get_dataset(imfolder,maskfolder):
    imp = SharedImageMaskImporter(data_path=imfolder,labels_path=maskfolder)
    dataset = fo.Dataset.from_importer(imp)
    four.random_split(dataset, {"train": 0.8, "val": 0.2})
    # from IPython import embed; embed()
    return dataset


    

def get_meanstd(dataset:fo.Dataset)->tuple[float,float]:
    if "data_folder" in dataset.info:
        meanstdpath = Path(dataset.info["data_folder"])/"meanstd.json"
    else:
        meanstdpath = None

    if meanstdpath is not None and meanstdpath.exists():
        with open(meanstdpath,"r") as f:
            MEAN,STD = json.load(f)
    else:
        from running_stats import RunningStats
        def get_im_mean_std(dataset:fo.Dataset):
            stats = RunningStats(n=np.array(0),m=np.array(0),s=np.array(0))
            for samp in dataset.iter_samples(progress=True):
                im = imread(samp.filepath)

                #image transforms - obviated by segmentation pipeline
                im = rescale_intensity(im)
                if im.dtype != np.uint8:
                    im = np.astype(im/255,np.uint8)
                
                stats += im;

            return stats.mean/255, stats.std/255 #from 0-1!!
        MEAN,STD = get_im_mean_std(dataset)
        print("Calculated dataset stats. mean:",MEAN,", std: ",STD)
        if meanstdpath is not None:
            with open(meanstdpath,"w") as f:
                json.dump((MEAN,STD),f)
    return MEAN,STD


if __name__ == "__main__":
    data_path = Path("datasets/optotaxis")
    imfolder = data_path/"images";#'C:\\Users\\miner\\OneDrive - University of North Carolina at Chapel Hill\\Bear Lab\\optotaxis calibration\\data\\segmentation_iteration_testing\\iter4\\round1\\images'
    maskfolder = data_path/"masks";#Path('C:/Users/miner/OneDrive - University of North Carolina at Chapel Hill/Bear Lab/optotaxis calibration/data/segmentation_iteration_testing/iter4/round1/masks')

    dataset = get_dataset(imfolder,maskfolder)
    mean,std = get_meanstd(data_path,dataset)

    
