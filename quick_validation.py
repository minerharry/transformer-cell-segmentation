import os
from pathlib import Path

from imageio import imread
import numpy as np
from tqdm import tqdm
from other_models import SegFormerModel, OneFormerModel
from train_mask2former import Mask2FormerModel
from segmentation_training import segment_data,process_presets
from segment_movie import make_deprocess_fn,make_process_fn,SegmentationParams

if __name__ == "__main__":

    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("validation.log"),
                            logging.StreamHandler()
                        ])
    # for dataset in ["optotaxis","optotaxis_half","optotaxis_quarter","stardist_10x","stardist_20x"]:
    if True:
        dataset = "optotaxis"
        setting = "3stack_split"
        # preset = process_presets[dataset][setting]
        preset = SegmentationParams(stack_images=True,do_splitting=False,stack_duplicate_missing=True)
        proc = make_process_fn(preset,"images")
        deproc = make_deprocess_fn(preset,"masks")

        # if "optotaxis" in dataset and "3stack" in setting:
        #     inp = Path("datasets")/dataset/"validation"/"wide_images"
        # else:
        #     inp = Path("datasets")/dataset/"validation"/"images"

        # outp = Path("datasets")/dataset/"validation"/"output_masks"
        # for f in outp.glob("*"):
        #     os.remove(f)

        inp = Path(r"C:\Users\miner\OneDrive - University of North Carolina at Chapel Hill\Senior Year\Spring 2025\Comp 590\53_mov6")
        outp = Path(r"C:\Users\miner\OneDrive - University of North Carolina at Chapel Hill\Senior Year\Spring 2025\Comp 590\53_mov6_masks")

        modeltype = SegFormerModel;
        modelkey = "segformer"
        modelname = f"{modelkey}_{setting}_{dataset}"

        print("model name:",modelname)

        segment_data(modeltype,modelname,inp,outp,proc,deproc)


        # from pathlib import Path

        # valp = Path("datasets")/dataset/"validation"/"masks"

        # IoUs = []
        # for f in tqdm(outp.glob("*.tif")):
        #     im1 = imread(outp/f.name)
        #     im2 = imread(valp/f.name)

        #     im1 = ~np.isclose(im1,0)
        #     im2 = ~np.isclose(im2,0)

        #     iou = np.sum(im1 * im2) / np.sum(im1 | im2)
        #     IoUs.append(iou)

        # mean = np.mean(IoUs)
        # logging.info(f"{dataset} {setting} {modelkey}, {len(IoUs)} masks")
        # logging.info(f"mean IoU: {mean}")
