## MODEL COMMANDS
K-Net uses openlab mim to do everything. Train and test via mim:
https://github.com/open-mmlab/mim/tree/main

```python
from mim import train

#repo determines the github repo. mmseg is semantic segmentation, mmdet can do panotpic (HOW TO SELECT?)
#eval=mIoU for semantic, pq for panoptic
train(package='mmseg', config='resnet18_8xb16_cifar10.py', gpus=0,
      other_args=('--work-dir', 'tmp','--eval','mIoU'))

## TO FINE-TUNE: I think you use load_from as a key *in the config*? 
#currently testing with the basic script they use

```

Mask2former uses detectron2. It seems like its https://github.com/facebookresearch/Mask2Former/blob/main/train_net.py ~~accepts a "resume" parameter somewhere - is that what I need?~~
-- no, resume is just about the training schedule. I want to specify its model weights.
-- weights are in cfg.MODEL.WEIGHTS. cfg is created in setup(args) which calls get_cfg, adding maskformer2_config and merging from user-supplied cfg file
-- OH I'm dumb they're already part of the weights. Now how to make the model not train from scratch?

