import json
import os
from typing import List

import torch
from torch import nn
from TorchCRF import CRF
from transformers import AutoConfig, AutoModel, AutoTokenizer


MODEL_DIR = r"result/roberta/final_CRF"


class TransformerCRF(nn.Module):
    def __init__(self, model_name: str, num_labels: int, use_focal_loss: bool = False, gamma: float = 2.0):
        super().__init__()
        self.num_labels = num_labels
        self.use_focal_loss = use_focal_loss
        self.gamma = gamma

        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            mask = attention_mask.bool()
            safe_labels = torch.where(labels == -100, torch.tensor(0, device=labels.device), labels)
            loss = -self.crf(logits, tags=safe_labels, mask=mask, reduction="mean")

        decoded_tags = self.crf.decode(logits, mask=attention_mask.bool())

        fake_logits = torch.zeros_like(logits)
        for i, tags in enumerate(decoded_tags):
            for j, tag in enumerate(tags):
                fake_logits[i, j, tag] = 1.0

        return {"loss": loss, "logits": fake_logits} if loss is not None else {"logits": fake_logits}


def load_labels(model_dir: str):
    with open(os.path.join(model_dir, "labels.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    id2label_raw = data["id2label"]
    label2id = data["label2id"]

    # keys trong json thường là string
    id2label = {int(k): v for k, v in id2label_raw.items()}
    return id2label, label2id


def load_model_and_tokenizer(model_dir: str):
    id2label, label2id = load_labels(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model = TransformerCRF(
        model_name="roberta-base",
        num_labels=len(label2id),
        use_focal_loss=False
    )

    state_dict = torch.load(
        os.path.join(model_dir, "pytorch_model.bin"),
        map_location="cpu"
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, tokenizer, id2label


def predict_tokens(tokens: List[str], model, tokenizer, id2label):
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"]
        )

    pred_ids = outputs["logits"].argmax(dim=-1)[0].tolist()
    word_ids = encoded.word_ids(batch_index=0)

    results = []
    previous_word_id = None

    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id == previous_word_id:
            continue

        label_id = pred_ids[token_idx]
        label = id2label[label_id]
        results.append((tokens[word_id], label))
        previous_word_id = word_id

    return results


if __name__ == "__main__":
    model, tokenizer, id2label = load_model_and_tokenizer(MODEL_DIR)

    samples = [
        ["Senior", "Python", "Developer", "in", "Hanoi", "with", "3", "years", "experience", "salary", "2000", "USD"],
        ["Java", "engineer", "needed", "in", "Da", "Nang"],
        ["5", "years", "of", "React", "experience"]
    ]

    for idx, sample_tokens in enumerate(samples, 1):
        print(f"\nSample {idx}:")
        predictions = predict_tokens(sample_tokens, model, tokenizer, id2label)
        for token, label in predictions:
            print(f"{token:15s} -> {label}")