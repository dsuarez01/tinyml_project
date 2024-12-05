import torch

from src.model import (
	load_llama_model,
	load_quantized_model,
	fine_tune_model,
)

from src.data import (
	load_tokenizer,
	load_dataset,
)

from src.metrics import (
	compute_perplexity,
)

def tune_quantized(num_steps, batch_size, model_path, quantized_params_path, dataset_path, device):
	# load tokenizer and dataset
	tokenizer = load_tokenizer(model_path)
	tune_dataloader = load_dataset(dataset_path, tokenizer, "train", batch_size)
	test_dataloader = load_dataset(dataset_path, tokenizer, "test", batch_size)
	# load quantized model, set to eval mode, move to device
	print("Loading original model...")
	model = load_llama_model(model_name=model_path, dtype="float32")
	
	print("Loading quantized weights...")
	quantized_params = torch.load(quantized_params_path)
	model = load_quantized_model(model, quantized_params)
	
	model.eval()
	model.to(device)

	# tune model and obtain perplexity after tuning
	print("Tuning model to recover perplexity...")
	model = fine_tune_model(model, tune_dataloader, tokenizer, device, num_steps)
	model.eval()

	quantized_perplexity = compute_perplexity(model, tokenizer, test_dataloader, device)
	print(f"Quantized Model Perplexity: {quantized_perplexity}")

	# save quantized model
	print("Saving quantized model...")
	save_quantized_model(quantized_params, f"./quantized_models/open_llama_3b_tuned_{num_steps}.pth")


if __name__ == '__main__':
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model_path = "./orig_models/open_llama_3b"
	quantized_params_path = "./quantized_models/open_llama_3b.pth"
	dataset_path = "./datasets/wikitext-2"
	num_steps = 100
	batch_size = 1

	tune_quantized(num_steps, batch_size, model_path, quantized_params_path, dataset_path, device)