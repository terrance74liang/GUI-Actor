import torch
from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer

CKPT_DIR = "/home/teliang/scratch/GUI-Actor/checkpoints/qwen25vl_sft" 
OUT_PATH = "/home/teliang/scratch/GUI-Actor/checkpoints/qwen25vl_sft/multi_patch_pointer_head.pt"

# Load the full model on CPU, no DeepSpeed here
model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
    CKPT_DIR,
    torch_dtype=torch.bfloat16,
    device_map="cpu",   # or omit device_map in older HF
)

# Extract only the multi-patch head weights
mph_state = model.multi_patch_pointer_head.state_dict()

# Save them
torch.save(mph_state, OUT_PATH)
print("Saved multi_patch_pointer_head weights to", OUT_PATH)
