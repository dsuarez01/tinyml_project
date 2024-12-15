from datasets import load_dataset, load_from_disk

## load huggingface dataset
def hf_load_dataset(path, name, split): # compute node online
    return load_dataset(path=path, name=name, split=split)

def hf_load_from_disk(path, split): # compute node offline, need to download dataset
    dataset = load_from_disk(path)
    if isinstance(dataset, dict): # multiple splits
        dataset = dataset[split]
    return dataset