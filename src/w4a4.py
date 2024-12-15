from functools import partial

import torch
from torch import nn

# core quantization method (simulated quantization)
def pseudo_quantize_tensor(
    tensor, n_bit=4, gp_size=-1
):
    original_shape = tensor.shape
    if gp_size > 0:
        assert original_shape[-1] % gp_size == 0
        tensor = tensor.reshape(-1, gp_size)

    assert tensor.dim() == 2

    # max and min vals in tensor (needed to calculate scale factor, zero pt)
    max_val = tensor.amax(dim=1, keepdim=True)
    assert max_val.dim() == 2 and max_val.size(0) == tensor.size(0) and max_val.size(1) == 1
    min_val = tensor.amin(dim=1, keepdim=True)
    assert min_val.dim() == 2 and min_val.size(0) == tensor.size(0) and min_val.size(1) == 1

    # calc scale factor and zero pt
    max_int = 2**n_bit - 1
    scales = (max_val-min_val).clamp(min=1e-5)/max_int
    assert scales.shape == max_val.shape
    zeros = (-torch.round(min_val/scales)).clamp_(0, max_int)
    assert scales.shape == min_val.shape

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(tensor).sum() == 0

    # map vals in the range [min, max] to lie within [0, 2**n_bit - 1]
    tensor = torch.clamp(torch.round(tensor/scales)+zeros, 0, max_int)
    assert tensor.dim() == 2 and tensor.size(0) == scales.size(0) and tensor.size(1) == gp_size

    # dequantize tensor (because this is pseudo-quantization)
    tensor = (tensor - zeros) * scales
    assert tensor.dim() == 2 and tensor.size(0) == scales.size(0) and tensor.size(1) == gp_size

    assert torch.isnan(tensor).sum() == 0

    tensor = tensor.reshape(original_shape)
    return tensor

@torch.no_grad()
def pseudo_quantize_model_salient_weight_fp16(
    model, w_bit, q_gp_size, input_act_stats, frac_salient,
):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_act_stats[n]).float()

            # find <frac_salient*100>% of the salient weight channels via importance
            num_salient_channels = int(frac_salient*m.weight.shape[1])
            if num_salient_channels == 0: # skip quantization if not needed
                continue
            _, salient_channels = torch.topk(importance, num_salient_channels)
            assert salient_channels.dim() == 1

            salient_weights = m.weight.data[..., salient_channels].clone() # back up the vals of salient weight channels
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, gp_size=q_gp_size) # pseudo quantization
            m.weight.data[..., salient_channels] = salient_weights # restore <frac_salient*100>% salient weight channels to orig FP16 vals

@torch.no_grad()
def register_pseudo_quant_act_hooks_fp16(
    model, a_bit, q_gp_size, input_act_stats, frac_salient,
):
    def quantize_acts(m, x, y, importance):
        if isinstance(x, tuple):
          x = x[0]

        # find <frac_salient*100>% of the salient activation channels via importance
        num_salient_channels = int(frac_salient*x.shape[-1])
        if num_salient_channels == 0: # skip quantization if not needed
            return
        _, salient_channels = torch.topk(importance, num_salient_channels)
        assert salient_channels.dim() == 1

        salient_acts = x.data[..., salient_channels].clone() # back up the vals of salient activation channels
        x.data = pseudo_quantize_tensor(x.data, n_bit=a_bit, gp_size=q_gp_size) # pseudo quantization
        x.data[..., salient_channels] = salient_acts # restore <frac_salient*100>% salient weight channels to orig FP16 vals

    # note: the hooks aren't removed here since activations are quantized dynamically (each forward pass)
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.register_forward_hook(
                partial(quantize_acts, importance=sum(input_act_stats[n]).float())
            )