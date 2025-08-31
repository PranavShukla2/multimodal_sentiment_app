import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel

class MultimodalSentimentClassifier(nn.Module):
    """
    A multimodal sentiment classifier that processes both image and text data.
    """
    def __init__(self, num_classes):
        super(MultimodalSentimentClassifier, self).__init__()
        
        # --- Image Branch ---
        # Use a pre-trained ResNet-50 model and remove the final classification layer
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer
        
        # --- Text Branch ---
        # Use a pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # --- Fusion and Classification Layers ---
        # The input features to the classifier will be the concatenated features
        # from both ResNet (2048) and BERT (768)
        self.fusion_dim = num_ftrs + self.bert.config.hidden_size # 2048 + 768 = 2816
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids=None, attention_mask=None, image=None):
        """
        Forward pass for the model. Handles unimodal (text or image) and multimodal inputs.
        """
        # --- Process Image Input ---
        image_features = None
        if image is not None:
            image_features = self.resnet(image) # Shape: (batch_size, 2048)

        # --- Process Text Input ---
        text_features = None
        if input_ids is not None:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Use the [CLS] token's hidden state as the sentence representation
            text_features = outputs.pooler_output # Shape: (batch_size, 768)
            
        # --- Fusion Logic ---
        if image_features is not None and text_features is not None:
            # If both modalities are present, concatenate their features
            combined_features = torch.cat((image_features, text_features), dim=1)
        elif image_features is not None:
            # If only image is present, pad text features with zeros
            text_padding = torch.zeros(image_features.shape[0], self.bert.config.hidden_size).to(image_features.device)
            combined_features = torch.cat((image_features, text_padding), dim=1)
        elif text_features is not None:
            # If only text is present, pad image features with zeros
            image_padding = torch.zeros(text_features.shape[0], 2048).to(text_features.device)
            combined_features = torch.cat((image_padding, text_features), dim=1)
        else:
            raise ValueError("At least one of image or text input must be provided.")
            
        # --- Classification ---
        logits = self.classifier(combined_features)
        return logits

