import torch
import torch.nn as nn

# two 4-bit values -> one 8-bit value
def pack_4bit_to_uint8(quantized_weights):
	quantized_weights = quantized_weights.clamp(0, 15).to(torch.uint8)
	flat_weights = quantized_weights.view(-1)
	packed_weights = flat_weights[::2] | (flat_weights[1::2] << 4)
	return packed_weights

# one 8-bit value -> two 4-bit values
def unpack_uint8_to_4bit(packed_weights):
	low_bits = packed_weights & 0x0F
	high_bits = (packed_weights >> 4) & 0x0F
	return torch.cat([low_bits, high_bits], dim=0)

# saliency-aware weight quantization
def quantize_weights(weights, num_bits=4, group_size=16, frac=0.2):
	device = weights.device
	weights_flat = weights.view(-1)
	num_groups = (weights_flat.size(0)+group_size-1)//group_size

	# weights divisible by group_size
	padded_size = num_groups * group_size
	weights_padded = torch.nn.functional.pad(weights_flat, (0, padded_size - weights_flat.size(0)))
	groups = weights_padded.view(num_groups, group_size)

	# activation magnitudes -> identify salient weights
	activation_magnitudes = groups.abs().mean(dim=1)
	num_protected = int(frac * num_groups)
	protected_indices = activation_magnitudes.topk(num_protected, largest=True).indices

	# group min and max
	min_vals, max_vals = groups.min(dim=1).values, groups.max(dim=1).values
	scale = torch.where(max_vals != min_vals, (max_vals-min_vals)/(2 ** num_bits - 1), torch.tensor(1e-6, device=device))
	zero_point = torch.round(-min_vals/scale)

	# quantize weights to 4-bit
	quantized_groups = torch.round((groups-min_vals.unsqueeze(1))/scale.unsqueeze(1) + zero_point.unsqueeze(1))
	quantized_groups = torch.clamp(quantized_groups, 0, 2 ** num_bits - 1).to(torch.uint8)

	# protect salient weights
	for idx in protected_indices:
		quantized_groups[idx] = groups[idx]

	# pack into 8-bit
	packed_weights = pack_4bit_to_uint8(quantized_groups.view(-1))
 
	del weights_flat, weights_padded, groups, min_vals, max_vals, quantized_groups # freeing intermediate tensors
	torch.cuda.empty_cache()

	return packed_weights, scale.to(device), zero_point.to(device), protected_indices

# dequantize weights for inference
def dequantize_weights(packed_weights, scale, zero_point, group_size=16, protected_indices=None):
	device = packed_weights.device
	unpacked_weights = unpack_uint8_to_4bit(packed_weights)
	num_groups = len(scale)
	unpacked_weights = unpacked_weights.view(num_groups, group_size).float()
	dequantized_weights = (unpacked_weights-zero_point.unsqueeze(1))*scale.unsqueeze(1)

	# restore salient weights
	if protected_indices is not None:
		for idx in protected_indices:
			dequantized_weights[idx] = unpacked_weights[idx]
	
	del unpacked_weights, scale, zero_point # free intermediate tensors
	torch.cuda.empty_cache()

	return dequantized_weights.view(-1)[:packed_weights.numel() * 2]

# quantizing linear layers (saliency-aware)
def quantize_linear_layers(model, num_bits=4, group_size=16, frac=0.2):
	quantized_params = {}
	for name, module in model.named_modules():
		if isinstance(module, nn.Linear):
			print(f"Quantizing weights for {name}...")
			with torch.no_grad():
				packed_weights, scale, zero_point, protected_indices = quantize_weights(module.weight, num_bits, group_size, frac)
				quantized_params[name] = (packed_weights, scale, zero_point, protected_indices)

				module.weight.data.copy_(
					dequantize_weights(packed_weights, scale, zero_point, group_size, protected_indices).view_as(module.weight)
				)

				del packed_weights, scale, zero_point, protected_indices # free intermediate tensors/memory
				torch.cuda.empty_cache()

	print("Linear layer weight quantization done.")
	return model, quantized_params

# dynamically quantize activations
def quantize_activations(activations, num_bits=4, group_size=16):
	device = activations.device
	activations_flat = activations.view(-1)
	num_groups = (activations_flat.size(0)+group_size-1)//group_size

	# padding of activations
	padded_size = num_groups * group_size
	activations_padded = torch.nn.functional.pad(activations_flat, (0, padded_size-activations_flat.size(0)))
	groups = activations_padded.view(num_groups, group_size)

	# min + max for each group
	min_vals, max_vals = groups.min(dim=1).values, groups.max(dim=1).values
	scale = torch.where(max_vals != min_vals, (max_vals-min_vals)/(2 ** num_bits - 1), torch.tensor(1e-6, device=device))
	zero_point = torch.round(-min_vals/scale)

	# Quantize activations
	quantized_groups = torch.clamp(
		torch.round((groups-min_vals.unsqueeze(1))/scale.unsqueeze(1) + zero_point.unsqueeze(1)),
		0,
		2 ** num_bits - 1,
	).to(torch.uint8)

	# Free intermediate tensors
	del activations_flat, activations_padded, groups, min_vals, max_vals
	torch.cuda.empty_cache()

	return quantized_groups, scale, zero_point

# quantization hook for activations
def quantize_hook(module, inputs, outputs):
	if isinstance(outputs, torch.Tensor):
		quantized, _, _ = quantize_activations(outputs)
		del outputs  # free intermediate memory
		torch.cuda.empty_cache()
		return quantized
	elif isinstance(outputs, (tuple, list)): # TO-DO: may not be needed?
		return type(outputs)(
			quantize_activations(t)[0] if isinstance(t, torch.Tensor) else t for t in outputs
		)
	return outputs

# apply activation quantization hooks
def apply_activation_hooks(model):
	for module_name, module in model.named_modules():
		if isinstance(module, nn.Linear):
			module.register_forward_hook(quantize_hook)
	print("Activation quantization hooks applied.")
	return model