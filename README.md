# tinyml_project
Repository for TinyML project (course # is 6.5940).

We explore mixed-precision weight 4-bit, activation 4-bit quantization (W4A4) for LLM compression and acceleration. 

A push-the-button script is included below for ease of use. Note that the SLURM environment that I am currently developing this project in uses a slightly different workflow.

## Push-the-button script: 

To use mixed-precision W4A4 quantization on the model, run the following command:

```
python -u app.py \
    --frac_salient_weights <fraction of salient weights preserved> \
    --plot_name <name of plot> \
    --model_path <path to model> \
    --eval_ds_path <path to eval. dataset> \
    --calib_ds_path <path to calib dataset>
```

Arguments:
- `--frac_salient_weights`: Fraction of salient weights to preserve during quantization (e.g., `0.01` for preserving 1% of weights).
- `--plot_name`: Name for the output plot (e.g., `opt-2.7b`). Plot saved in the `./plots/` directory.
- `--model_path`: Path to the pretrained model (e.g., `./models/opt-2.7b`). Model should be pre-downloaded.
- `--eval_ds_path`: Path to the evaluation dataset (e.g., `./datasets/wikitext-2`). Dataset should be pre-downloaded.
- `--calib_ds_path`: Path to the calibration dataset (e.g., `./datasets/pile-val-backup`). Dataset should also be pre-downloaded.
