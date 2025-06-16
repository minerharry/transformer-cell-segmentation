from pathlib import Path
from segmentation_training import segment_data, stardist_20x_enum_regkey, process_presets
from segment_movie import make_deprocess_fn, make_process_fn
from seqformer import SeqFormerModel

if __name__ == "__main__":
    pre = process_presets["stardist_20x"]["nostack_split"]
    proc = make_process_fn(pre,"images")
    deproc = make_deprocess_fn(pre,"masks")
    
    segment_data(SeqFormerModel,
                 r"C:\Users\miner\Documents\Github\SeqFormer\seq_test",
                 "datasets/stardist_20x/validation/images",
                 "datasets/stardist_20x/validation/output_masks",
                 proc,deproc,
                 model_kwargs={"regkey":stardist_20x_enum_regkey},
                 as_dataset=True)