import argparse
import torch
from src.model import quantize_linear_layers, fine_tune_model
from src.data import load_dataset, load_tokenizer
from src.metrics import compute_perplexity

def main(args):
    print("Loading tokenizer and dataset...") # load tokenizer and dataset
    tokenizer = load_tokenizer(args.model_path)
    train_dataloader = load_dataset(args.dataset_path, tokenizer, split="train", batch_size=args.batch_size)
    test_dataloader = load_dataset(args.dataset_path, tokenizer, split="test", batch_size=args.batch_size)

    print("Quantizing model...") # load and quantize model
    model = torch.load(args.model_path)
    model, quantized_params = quantize_linear_layers(model, num_bits=4, group_size=16)
    
    print(f"Fine-tuning model for {args.num_tune_steps} steps...") # fine-tune model
    model = fine_tune_model(model, train_dataloader, tokenizer, device="cuda", num_steps=args.num_tune_steps)
    
    print("Evaluating perplexity on test set...") # perplexity comp
    perplexity = compute_perplexity(model, tokenizer, test_dataloader, device="cuda")
    print(f"Final Perplexity: {perplexity}")
    
    print(f"Saving quantized model to {args.output_path}...") # quantized model, save
    torch.save({"model_state_dict": model.state_dict(), "quantized_params": quantized_params}, args.output_path)
    print("Quantized model saved successfully!")

if __name__ == "__main__":
    # we use argparse for the push-the-button script
    parser = argparse.ArgumentParser(description="Push-the-button script for W4A4 quantization + fine-tuning")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save quantized model and parameters")
    parser.add_argument("--num_tune_steps", type=int, default=500, help="Number of fine-tuning steps (default: 500)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation (default: 16)")
    args = parser.parse_args()

    main(args)