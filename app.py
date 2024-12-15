import argparse
import gc

import torch
from torch import nn
import matplotlib.pyplot as plt

from src.constants import Byte, KiB, MiB, GiB

from src.w4a4 import (
    pseudo_quantize_tensor, 
    pseudo_quantize_model_salient_weight_fp16, 
    register_pseudo_quant_act_hooks_fp16
)

from src.model import (
    load_tokenizer, 
    load_model, 
    get_model_size
)

from src.datasets import hf_load_from_disk

from src.eval_utils import (
    compute_perplexity, 
    process_calib_dataset, 
    get_calib_feat_with_outliers
)

def main(args):
    perplexities_fp16 = []
    perplexities_quantized = []

    # load calib and eval datasets
    calib_dataset = hf_load_from_disk(args.calib_ds_path, "validation")
    eval_dataset = hf_load_from_disk(args.eval_ds_path, "test")

    # load tokenizer + FP16 model
    tokenizer = load_tokenizer(args.model_path)
    model = load_model(args.model_path, torch_dtype=torch.float16)
    input_act_stats = get_calib_feat_with_outliers(model, tokenizer, calib_dataset) # collect input activation statistics

    model_perplexity = compute_perplexity(model, tokenizer, eval_dataset) # eval FP16 model
    perplexities_fp16.append(model_perplexity.cpu().item())
    model_size = get_model_size(model, data_width=16, group_size=128)
    print(f"\nFP16 model perplexity: {model_perplexity:.2f}")
    print(f"FP16 model size: {model_size/MiB:.2f} MiB")

    frac_salient_activations_range = [i / 10 for i in range(1, 10)]

    for frac_salient_activations in frac_salient_activations_range:

        # clear memory, reload model
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = load_model(args.model_path, torch_dtype=torch.float16)

        pseudo_quantize_model_salient_weight_fp16( # quantize weights
            model,
            w_bit=4,
            q_gp_size=128,
            input_act_stats=input_act_stats,
            frac_salient=args.frac_salient_weights,
        )

        register_pseudo_quant_act_hooks_fp16( # register activation quantization hooks
            model,
            a_bit=4,
            q_gp_size=128,
            input_act_stats=input_act_stats,
            frac_salient=frac_salient_activations,
        )

        # eval mixed-precision quantized model
        model_perplexity = compute_perplexity(model, tokenizer, eval_dataset)
        perplexities_quantized.append(model_perplexity.cpu().item())
        print(f"\nQuantized model (frac_salient_activations={frac_salient_activations:.2f}) perplexity: {model_perplexity:.2f}")

    # plotting
    plt.figure(figsize=(10, 6))
    plt.axhline(y=perplexities_fp16[0], color='r', linestyle='--', label='FP16 Model Perplexity')
    plt.plot(
        frac_salient_activations_range,
        perplexities_quantized,
        marker='o',
        label='W4A4 Mixed-Precision Quantized Model Perplexity',
    )
    plt.xlabel("Frac. of Salient Activation Channels Quantized")
    plt.ylabel("Perplexity")
    plt.title(f"Perplexity vs. Frac. Quantized Salient Activation Channels \nModel: {args.model_path}")
    plt.legend()
    plt.grid(True)
    
    plot_path = f"./plots/{args.plot_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # we use argparse for the push-the-button script
    parser = argparse.ArgumentParser(description="Push-the-button script for W4A4 AWQ")
    parser.add_argument("--frac_salient_weights", type=float, required=True, help="Fraction of salient weights preserved")
    parser.add_argument("--plot_name", type=str, required=True, help="Name for plot (to be saved in ./plots/)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model (should be pre-downloaded)")
    parser.add_argument("--eval_ds_path", type=str, required=True, help="Path to evaluation dataset (should be pre-downloaded)")
    parser.add_argument("--calib_ds_path", type=str, required=True, help="Path to calibration dataset (should be pre-downloaded)")

    args = parser.parse_args()

    main(args)