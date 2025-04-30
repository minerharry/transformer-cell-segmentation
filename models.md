# Datasets:
- https://github.com/CellMigrationLab/Datasets?tab=readme-ov-file#Image-Data


# Model search sources:
- https://github.com/lxtGH/Awesome-Segmentation-With-Transformer


## Phase 1: Image
### Datasets
- Just collect normal segmentation models capable of segmenting grayscale and 3color images. Run it through cell-segmentation.
- the two zenodo datasets have semantic and panoptic both. For phase 1 maybe just convert all to semantic? idk
- Datasets:
  - Brightfield:
    - https://zenodo.org/records/13304399 - 10x BF
      - MANY 3STACKS, MANY HAVE BEEN EXTRACTED TO THE QC FOLDER. REAGGREGATE AND TEST-TRAIN SPLIT
    - https://zenodo.org/records/10572122 - 20x BF, ???
      - 4 sets of 6 frames each. it might be worth using half the data as validation - very much a few-shot test
    - My dataset
  - Fluorescent: ...

### Architectures
- K-Net (https://arxiv.org/abs/2106.14855) https://github.com/ZwwWayne/K-Net/
  - Semantic, Panoptic
    - Make sure to only use the transformer variants for this project!
    - Instance doesn't use transformers at all. Also not really want we want so just ignore
      - the Swin-L in the panoptic model is a little scary - hopefully buster can handle it. Do it last!
- Mask2Former (https://arxiv.org/abs/2112.01527v3) https://bowenc0221.github.io/mask2former/
  - Semantic, Panoptic
    - 
- Vision Mamba

## Phase 2: Improved Image, Video
- Add panoptic segmentation to my dataset? probably won't be too terribly hard, just find all the instances of touching cells and maybe make some extra masks to deal with it idk.


Video K-Net: https://github.com/lxtGH/Video-K-Net, https://arxiv.org/abs/2204.04656
- Video Panoptic
- Video Segmantic
- Video Instance

TeVit:



    









- single-frame segmentation (1 channel or 3stacked channel)
-- segformer
-- PVT
-- mask2former
-- DETR
-- Deformable DETR

- video segmentation (autoregressive / clip)
-- mask2former-vis
-- Video k-net
-- TimeSFormer
-- VideoMae
-- MeMVit

- Object Detection and Tracking via basic LAP
--

- End-to-End Tracking
-- TrackFormer
-- Video K-net