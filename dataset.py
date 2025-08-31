import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class MultimodalDataset(Dataset):
    """
    Custom PyTorch Dataset for loading image-text pairs.
    
    Assumes the label file is a text file where each line contains:
    image_id,sentiment_label,text_caption
    e.g.: 12345.jpg,2,"A beautiful sunny day at the beach"
    """
    def __init__(self, data_dir, label_file, image_transform=None, tokenizer=None, max_length=128):
        self.data_dir = data_dir
        # Load the labels and text from the file
        self.df = pd.read_csv(label_file, header=None, names=['id', 'sentiment', 'text'])
        self.image_transform = image_transform
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image path, text, and label
        image_name = self.df.loc[idx, 'id']
        image_path = os.path.join(self.data_dir, str(image_name))
        text = self.df.loc[idx, 'text']
        label = int(self.df.loc[idx, 'sentiment'])

        # --- Image Processing ---
        try:
            image = Image.open(image_path).convert('RGB')
            if self.image_transform:
                image = self.image_transform(image)
        except (FileNotFoundError, UnboundLocalError):
            # If image is not found, create a placeholder tensor of zeros
            image = torch.zeros((3, 224, 224))

        # --- Text Processing ---
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }
