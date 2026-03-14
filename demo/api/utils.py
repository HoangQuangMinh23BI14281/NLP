import torch


def get_entities(tokens, tags, token_scores):
    """Extract entities from word-level tokens/tags and aggregate confidence scores."""
    entities = []
    current_entity = None

    for idx, (token, tag, score) in enumerate(zip(tokens, tags, token_scores)):
        if tag == "O":
            if current_entity:
                current_entity["score"] = round(sum(current_entity["_scores"]) / len(current_entity["_scores"]), 4)
                del current_entity["_scores"]
                entities.append(current_entity)
            current_entity = None
            continue

        if tag.startswith("B-"):
            if current_entity:
                current_entity["score"] = round(sum(current_entity["_scores"]) / len(current_entity["_scores"]), 4)
                del current_entity["_scores"]
                entities.append(current_entity)

            current_entity = {
                "type": tag[2:],
                "text": token,
                "start_idx": idx,
                "end_idx": idx,
                "_scores": [float(score)],
            }
        elif tag.startswith("I-") and current_entity and tag[2:] == current_entity["type"]:
            current_entity["text"] += " " + token
            current_entity["end_idx"] = idx
            current_entity["_scores"].append(float(score))
        elif "-" not in tag:
            # Some checkpoints emit plain entity tags (e.g., LOC) instead of BIO tags.
            if current_entity:
                current_entity["score"] = round(sum(current_entity["_scores"]) / len(current_entity["_scores"]), 4)
                del current_entity["_scores"]
                entities.append(current_entity)

            current_entity = {
                "type": tag,
                "text": token,
                "start_idx": idx,
                "end_idx": idx,
                "_scores": [float(score)],
            }
        else:
            if current_entity:
                current_entity["score"] = round(sum(current_entity["_scores"]) / len(current_entity["_scores"]), 4)
                del current_entity["_scores"]
                entities.append(current_entity)
            current_entity = None

    if current_entity:
        current_entity["score"] = round(sum(current_entity["_scores"]) / len(current_entity["_scores"]), 4)
        del current_entity["_scores"]
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

        logits = outputs["logits"] if isinstance(outputs, dict) and "logits" in outputs else outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        
        if "decoded_tags" in outputs:
            pred_ids = outputs["decoded_tags"][0]
        else:
            pred_ids = torch.argmax(logits, dim=2)[0].cpu().numpy().tolist()
            
        # Group subwords into words
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        final_tokens = []
        final_labels = []
        final_scores = []
        
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
                final_scores.append(float(probs[i, label_id].item()))
            else:
                # It's a subword of the same word (e.g., ##per)
                # Append to the last token and skip its label
                subword = tokens[i].replace("##", "")
                final_tokens[-1] += subword
            
            last_word_idx = word_idx
        
        return {
            "tokens": final_tokens,
            "tags": final_labels,
            "token_scores": [round(score, 4) for score in final_scores],
            "entities": get_entities(final_tokens, final_labels, final_scores),
        }
