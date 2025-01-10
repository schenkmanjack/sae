from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from datasets import Dataset
from torch.utils.data import TensorDataset, DataLoader

# Tokenize the dataset
def tokenize_function(tokenizer, texts):
    return tokenizer(
        texts,
        padding=True,  # Now padding works without errors
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

def prepare_text_data(model_name='gpt2', num_samples=4, batch_size=16):
    """Prepares the text data and tokenizes.
    
    Args:
    model_name: str, the model name to use for tokenization
    num_samples: int, the number of samples to use
    batch_size: int, the batch size for the DataLoader
    
    Returns:
    dataloader: the DataLoader object for the text data
    model: the LLM model
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set EOS token as the padding token
    # Alternatively: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # If you add a special pad token: model.resize_token_embeddings(len(tokenizer))

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    # dataset = load_dataset('bookcorpus', split='train', trust_remote_code=True)
    # texts = dataset['text'][:num_samples]

    dataset = load_dataset('bookcorpus', split='train', trust_remote_code="True", streaming=True)
    # Efficiently fetch only the first 1000 samples
    texts = []
    num_samples = 1000

    for i, record in enumerate(dataset):
        if i >= num_samples:  # Stop once we've fetched the required number of samples
            break
        texts.append(record['text'])
    
    # Create a Dataset object
    dataset_texts = Dataset.from_dict({"text": texts})
    


    encoded_inputs = tokenize_function(tokenizer, texts)  # This should now work without errors

    # tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4)  # Use 4 processes for tokenization
    # encoded_inputs = tokenized_dataset['input_ids']

    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_static = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # return dataloader, texts, model
    return dataloader, dataloader_static, dataset_texts, tokenizer, model

