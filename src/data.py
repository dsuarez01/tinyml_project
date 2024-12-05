from transformers import AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader

# to load tokenizer, note pad token handling
def load_tokenizer(model_path):
	
	tokenizer = AutoTokenizer.from_pretrained(model_path)

	if tokenizer.pad_token is None: # set pad token if not already set
		tokenizer.pad_token = tokenizer.eos_token

	return tokenizer

# dataset preprocessing
def load_dataset(dataset_path, tokenizer, split, batch_size=16):
	dataset_dict = load_from_disk(dataset_path)
	dataset = dataset_dict[split]

	def tokenize_fn(examples):
		return tokenizer(examples["text"], padding="max_length", truncation=True)

	tokenized_dataset = dataset.map(tokenize_fn, batched=True)
	tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
	dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
	return dataloader