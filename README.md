# tinyml_project
Repository for TinyML project (course # is 6.5940).

We explore weight 4-bit, activation 4-bit quantization (W4A4) for LLM compression and acceleration. 

A push-the-button script is in progress, as the SLURM environment that I am currently developing this project in uses a slightly different workflow.

## Push-the-button script [in progress]: 

To use W4A4 quantization on the model, run the following command:

```
python app.py \
    --model_path ./models/open-llama-7B \
    --dataset_path ./datasets/wikitext-2 \
    --num_tune_steps 500 \
    --batch_size 16 \
    --output_path ./quantized_results
```

Arguments:
    --model_path: Path to the pretrained model (e.g., ./models/open-llama-7B).
    --dataset_path: Path to the dataset (e.g., ./datasets/wikitext-2).
    --num_tune_steps: Number of fine-tuning steps (default: 500).
    --batch_size: Batch size for training and evaluation (default: 16).
    --output_path: Path to save the quantized model and parameters.