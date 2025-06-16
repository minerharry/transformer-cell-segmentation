

from pathlib import Path
from train_huggingface import HuggingFaceModel
from transformers.models.oneformer import OneFormerForUniversalSegmentation, OneFormerImageProcessor
from transformers.models.segformer import SegformerForSemanticSegmentation, SegformerImageProcessor
from transformers import BatchFeature
#I think this just means get whatever the default is for hugging face?
OneProcessor = OneFormerImageProcessor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False,class_info_file=str(Path("oneformer_classes.json").absolute()))

class OneFormerModel(HuggingFaceModel):
    @classmethod
    def shortname(cls):
        return "oneformer"
    
    @classmethod
    def load_type(cls):
        return OneFormerForUniversalSegmentation
    
    def pre_process_semantic_segmentation(self, images, **kwargs) -> BatchFeature:
        return OneProcessor(images, **kwargs)
    
    def post_process_semantic_segmentation(self, outputs, target_sizes: list[tuple] | None = None) -> list:
        return OneProcessor.post_process_semantic_segmentation(outputs, target_sizes)
    

SegProcessor = SegformerImageProcessor(reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)
class SegFormerModel(HuggingFaceModel):
    pretrained = "nvidia/mit-b0"
    @classmethod
    def shortname(cls):
        return "segformer"

    @classmethod
    def load_type(cls):
        return SegformerForSemanticSegmentation
    
    def pre_process_semantic_segmentation(self, images, **kwargs) -> BatchFeature:
        return SegProcessor(images,**kwargs)
    
    def post_process_semantic_segmentation(self, outputs, target_sizes: list[tuple] | None = None) -> list:
        return SegProcessor.post_process_semantic_segmentation(outputs,target_sizes=target_sizes);
        

