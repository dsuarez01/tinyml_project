import torch

from src.model import (
	load_llama_model,
	save_quantized_model,
	load_quantized_model,
)

from src.w4a4_helper import (
	quantize_linear_layers,
	apply_activation_hooks,
)

def quantize(device):
	model = load_llama_model(model_name="./orig_models/open_llama_3b", dtype="float32")
	model.to("cuda")

	print("Starting weight quantization...")
	model, quantized_params = quantize_linear_layers(model)

	print("Applying activation hooks...")
	quantized_model = apply_activation_hooks(model)

	print("Saving quantized model...")
	save_quantized_model(quantized_params, "./quantized_models/open_llama_3b.pth")
	print("Quantized model weights, parameters saved.")

if __name__ == '__main__':
	device = "cuda" if torch.cuda.is_available() else "cpu"
	quantize(device)