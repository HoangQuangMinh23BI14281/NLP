import torch
from torch import nn
from torchcrf import CRF
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification

class TransformerCRF(nn.Module):
    def __init__(self, model_name_or_path, num_labels, use_focal_loss=False, gamma=2.0):
        super(TransformerCRF, self).__init__()
        self.num_labels = num_labels
        self.use_focal_loss = use_focal_loss
        self.gamma = gamma
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        # Prefer pretrained backbone weights from model folder when available.
        try:
            self.transformer = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        except Exception:
            self.transformer = AutoModel.from_config(self.config)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            mask = attention_mask.bool()
            # Ensure labels are within valid range for CRF
            safe_labels = torch.where(labels == -100, torch.tensor(0, device=labels.device), labels)
            
            if self.use_focal_loss:
                # Sequence-level Focal Loss implementation as per training script
                nll = -self.crf(logits, tags=safe_labels, mask=mask, reduction='none')
                pt = torch.exp(-nll)
                loss = (((1 - pt) ** self.gamma) * nll).mean()
            else:
                # Standard CRF NLL Loss
                loss = -self.crf(logits, tags=safe_labels, mask=mask, reduction='mean')

        # Decoding
        eval_mask = attention_mask.bool()
        decoded_tags = self.crf.decode(logits, mask=eval_mask)
        
        # Create fake logits for consistency with Trainer/Inference expected format
        # or we can just return decoded tags
        return {"loss": loss, "logits": logits, "decoded_tags": decoded_tags}

def load_ner_model(model_path, arch_type="Base_CE"):
    """
    Loads the model based on architecture type.
    """
    if arch_type == "Base_CE":
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    elif "CRF" in arch_type:
        config = AutoConfig.from_pretrained(model_path)
        num_labels = len(config.id2label)
        model = TransformerCRF(model_path, num_labels=num_labels)
        # Load state dict if it's a custom saved model
        # For CRF, we usually save it as a pytorch_model.bin or similar
        # If using AutoModel inside, we might need to load manually
        state_dict_path = f"{model_path}/pytorch_model.bin" 
        # Checking for other common names
        import os
        if not os.path.exists(state_dict_path):
             state_dict_path = f"{model_path}/pytorch_model.pth"
        
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")
    
    model.eval()
    return model
