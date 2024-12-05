import torch
from src.model import (
	load_llama_model,
	load_quantized_model,
)
from src.data import (
	load_tokenizer,
)

def test_inference(model_path, quantized_params_path, inputs, device):
	# load model + tokenizer
	print("Loading original model...")
	model = load_llama_model(model_name=model_path, dtype="float32")
	tokenizer = load_tokenizer(model_path)

	# load quantized weights
	print("Loading quantized weights...")
	quantized_params = torch.load(quantized_params_path)
	model = load_quantized_model(model, quantized_params)
	model.eval()
	model.to(device)

	print("Quantized model loaded, starting inference...")

	# tokenize inputs
	encoded_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)

	# generate and decode outputs
	with torch.no_grad():
		outputs = model.generate(
			input_ids=encoded_inputs["input_ids"],
			max_new_tokens=50,
			do_sample=True,
			top_k=50,
			top_p=0.95,
		)
	decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
	return decoded_outputs

if __name__ == '__main__':
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model_path = "./orig_models/open_llama_3b"
	quantized_params_path = "./quantized_models/open_llama_3b.pth"
	inputs = [ # TO-DO: add more examples here
		"What is the capital of France?",
		"Who wrote the novel '1984'?",
		"Explain the significance of the Turing Test.",
	]

	results = test_inference(model_path, quantized_params_path, inputs, device)
	for idx, (inp, out) in enumerate(zip(inputs, results)):
		print(f"Input {idx + 1}: {inp}")
		print(f"Output {idx + 1}: {out}")
		print()