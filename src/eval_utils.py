from functools import partial

import torch
from torch import nn

# perplexity computation
def compute_perplexity(
    model, tokenizer, dataset,
):
    testenc = tokenizer("\n\n".join(dataset['text']), return_tensors='pt')
    testenc = testenc.input_ids.to(model.device)
    nsamples = 40
    model = model.eval()

    nlls = []
    for i in range(nsamples):
        batch = testenc[:, (i * 2048):((i + 1) * 2048)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * 2048):((i + 1) * 2048)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * 2048
        nlls.append(neg_log_likelihood)

    return torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))

## calibration dataset used to obtain activation statistics
# (dataset is a huggingface dataset, in our case assumed to be from MIT Han Lab)
# TO-DO: handle online vs offline compute nodes differently
def process_calib_dataset(
    dataset, tokenizer=None, n_samples=256, block_size=512
):
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > block_size:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break

    # cat all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]

@torch.no_grad()
def get_calib_feat_with_outliers(
    model, tokenizer, calib_dataset, n_samples=256, block_size=512,
):
    input_act_stats = dict()

    def stat_input_max_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        if name not in input_act_stats:
            input_act_stats[name] = [x_max]
        else:
            input_act_stats[name] += [x_max]

    # register hooks for all nn.Linear layers
    hooks = []
    for name, m in model.named_modules():
        # for activation statistics
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    partial(stat_input_max_hook, name=name)
                )
            )

    print("Collecting activation statistics...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # collect statistics based on calibration dataset
    samples = process_calib_dataset(calib_dataset, tokenizer, n_samples=n_samples, block_size=block_size)
    for input_ids in samples:
        input_ids = input_ids.to(device)
        model(input_ids)

    # remove hooks
    for hook in hooks:
        hook.remove()

    return input_act_stats