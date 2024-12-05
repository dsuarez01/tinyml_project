import torch

from src.model import (
	load_llama_model,
)

from src.data import (
	load_tokenizer,
	load_dataset,
)

from src.metrics import (
	compute_perplexity,
)

def perplexity_fp16(model_path, dataset_path, device):
	# load tokenizer and dataset
	tokenizer = load_tokenizer(model_path)
	dataset = load_dataset(dataset_path, tokenizer)

	# load FP16 model, set to eval mode, move to device
	print("Loading FP16 model...")
	model = load_llama_model(model_name=model_path, dtype="float16")	
	model.eval()
	model.to(device)

	perplexity = compute_perplexity(model, tokenizer, dataset, device)
	print(f"FP16 Model Perplexity: {perplexity}")


if __name__ == '__main__':
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model_path = "./orig_models/open_llama_3b"
	dataset_path = "./datasets/wikitext-2"

	perplexity_fp16(model_path, dataset_path, device)