# perplexity computation
def compute_perplexity(model, tokenizer, dataloader, device):
	assert not model.training, "Model is not in evaluation mode, call model.eval()."
	total_loss = 0.0
	total_tokens = 0

	for batch in dataloader:
		inputs = batch["input_ids"].to(device)
		attention_mask = batch["attention_mask"].to(device)

		with torch.no_grad():
			outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=inputs)
			loss = outputs.loss
			
			# exclude padding tokens from perplexity calc
			valid_tokens = attention_mask.sum().item()
			total_loss += loss.item() * valid_tokens
			total_tokens += valid_tokens

	perplexity = math.exp(total_loss / total_tokens)
	return perplexity