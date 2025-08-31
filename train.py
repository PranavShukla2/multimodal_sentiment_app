import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
import pandas as pd

# Import project-specific modules
from model import MultimodalSentimentClassifier
from dataset import MultimodalDataset
import config

def train_model():
    """
    Main function to handle the model training process.
    """
    # --- 1. Setup ---
    device = config.DEVICE
    print(f"Using device: {device}")

    # Image transformations
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # --- 2. Data Loading ---
    full_dataset = MultimodalDataset(
        data_dir=config.DATA_DIR,
        label_file=config.LABEL_FILE,
        image_transform=image_transform,
        tokenizer=tokenizer
    )

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    if train_size == 0 and len(full_dataset) > 0: train_size = 1 # Ensure at least one training sample
    val_size = len(full_dataset) - train_size
    
    # Handle tiny datasets
    if val_size == 0 and train_size > 1:
        train_size -= 1
        val_size = 1
        
    if len(full_dataset) > 1 :
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    else: # If only one sample, use it for both training and validation
        train_dataset = full_dataset
        val_dataset = full_dataset


    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Data loaded. Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # --- 3. Model, Optimizer, and Loss Function ---
    model = MultimodalSentimentClassifier(num_classes=config.NUM_CLASSES).to(device)
    
    # Optimizer (AdamW is recommended for transformer-based models)
    # CORRECTED LINE: Removed the 'correct_bias' argument
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Loss function for classification
    criterion = nn.CrossEntropyLoss()
    
    # --- 4. Training Loop ---
    epochs = 5 
    total_steps = len(train_loader) * epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    print("Starting training...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        if not train_loader: continue # Skip if no training data

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            model.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, image=images)
            
            loss = criterion(outputs, labels)
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        if not val_loader: continue # Skip if no validation data

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, image=images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

    # --- 5. Save the Trained Model ---
    torch.save(model.state_dict(), config.MODEL_WEIGHTS)
    print(f"Training complete. Model saved to {config.MODEL_WEIGHTS}")


if __name__ == '__main__':
    train_model()

