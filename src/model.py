import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from transformers import AutoModelForCausalLM

from .w4a4_helper import dequantize_weights


# loads llama model on available device
def load_llama_model(model_name, dtype):

	dtype_map = {"float16": torch.float16, "float32": torch.float32}

	model = AutoModelForCausalLM.from_pretrained(
		model_name,
		torch_dtype=dtype_map[dtype],
		device_map="auto",
	)
	return model

# save quantized parameters (model)
def save_quantized_model(quantized_params, save_path):
	torch.save(quantized_params, save_path)
	print(f"Quantized model saved to {save_path}.")

# load quantized model
def load_quantized_model(model, quantized_params):
	for name, module in model.named_modules():
		if name in quantized_params:
			packed_weights,scale,zero_point = quantized_params[name]
			# need to dequantize weights before copying back into model
			module.weight.data.copy_(dequantize_weights(packed_weights, scale,zero_point, group_size=16).view_as(module.weight))
	print("Quantized model loaded.")
	return model

# fine-tune model for num_steps, recover perplexity of FP16 model
def fine_tune_model(model, dataloader, tokenizer, device, num_steps=100):
	
	optimizer = AdamW(model.parameters(), lr=5e-6)  # TO-DO: maybe adjust the lr
	model.train()

	total_loss = 0.0

	# dataloader iteration for num_steps
	for step, batch in enumerate(dataloader):
		if step >= num_steps:
			break

		inputs = batch["input_ids"].to(device)
		attention_mask = batch["attention_mask"].to(device)

		# forward + backward pass
		optimizer.zero_grad()
		outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=inputs)
		loss = outputs.loss
		loss.backward()

		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # grad clipping to stabilize training
		
		optimizer.step()

		total_loss += loss.item()

		if step % 10 == 0:
			avg_loss = total_loss / (step + 1)
			print(f"Step {step}, Avg Loss: {avg_loss}")

		del inputs, attention_mask, outputs # freeing (GPU) memory
		torch.cuda.empty_cache()
		
	print("Fine-tuning complete.")
	return model