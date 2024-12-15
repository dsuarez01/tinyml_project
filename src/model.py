from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

## load huggingface tokenizer
def load_tokenizer(model_path): # to load tokenizer, includes pad token handling
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    if tokenizer.pad_token is None: # set pad token if not already set
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

def load_model(model_path, torch_dtype):
    return AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, device_map="auto")

# use to measure model size (might need constants in constants.py)
def get_model_size(
    model, data_width=16, group_size=-1
):
    if group_size != -1:
        data_width += (16 + 4)/group_size
    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements*data_width