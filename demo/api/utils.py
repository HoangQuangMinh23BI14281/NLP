import torch
from transformers import AutoTokenizer

def get_entities(tokens, tags):
    """
    Extracts entities from tokens and tags using BIO scheme.
    """
    entities = []
    current_entity = None
    
    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "type": tag[2:],
                "text": token,
                "start_idx": tokens.index(token) # Simplification
            }
        elif tag.startswith("I-") and current_entity:
            if tag[2:] == current_entity["type"]:
                current_entity["text"] += " " + token
            else:
                entities.append(current_entity)
                current_entity = None
        else:
            if current_entity:
                entities.append(current_entity)
            current_entity = None
            
    if current_entity:
        entities.append(current_entity)
        
    return entities

class NERInference:
    def __init__(self, model, tokenizer, id2label):
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, text):
        # inputs here is a BatchEncoding object
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        word_ids = inputs.word_ids(batch_index=0)
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        if "decoded_tags" in outputs:
            pred_ids = outputs["decoded_tags"][0]
        else:
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=2)[0].cpu().numpy().tolist()
            
        # Group subwords into words
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        final_tokens = []
        final_labels = []
        
        last_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            
            # If it's a new word
            if word_idx != last_word_idx:
                # Add the first subword and its label
                token = tokens[i]
                final_tokens.append(token)
                label_id = pred_ids[i]
                final_labels.append(self.id2label[str(label_id)] if str(label_id) in self.id2label else self.id2label[label_id])
            else:
                # It's a subword of the same word (e.g., ##per)
                # Append to the last token and skip its label
                subword = tokens[i].replace("##", "")
                final_tokens[-1] += subword
            
            last_word_idx = word_idx
        
        return {
            "tokens": final_tokens,
            "tags": final_labels,
            "entities": get_entities(final_tokens, final_labels)
        }
