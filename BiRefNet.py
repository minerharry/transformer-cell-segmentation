from pathlib import Path
from transformers.feature_extraction_utils import BatchFeature
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForImageSegmentation
from transformers.models.auto.processing_auto import AutoProcessor
from birefnet.models.birefnet import BiRefNet
from train_huggingface import HuggingFaceModel


# BiRefProcessor = AutoProcessor.from_pretrained("ZhengPeng7/BiRefNet",trust_remote_code=True);
class BiRefNetModel(HuggingFaceModel):
    pretrained = "ZhengPeng7/BiRefNet"

    def load_pretrained(self,Model:type[PreTrainedModel],modelpath:str|Path,gpu:bool,**kwargs):
        return Model.from_pretrained(modelpath)

    @classmethod
    def load_type(cls)->type[PreTrainedModel]:
        return BiRefNet
    
    def pre_process_semantic_segmentation(self, images, **kwargs) -> BatchFeature:
        from IPython import embed; embed()
        return BiRefProcessor(images,**kwargs)
        
    
    def post_process_semantic_segmentation(self, outputs, target_sizes: list[tuple] | None = None) -> list:
        from IPython import embed; embed()
        return BiRefProcessor.post_process_semantic_segmentation(outputs,target_sizes=target_sizes)
    
    @classmethod
    def shortname(cls) -> str:
        return "birefnet"