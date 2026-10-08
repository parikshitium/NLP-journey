from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


dataset = load_dataset('mteb/banking77')
data = dataset['test']
index = torch.load('semantic_index.pt')
embeds = index['embeddings']
texts = index['texts']
labels = index['labels']
name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name)

def get_embed(text):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    with torch.inference_mode():
        outputs = model(**inputs)
    mask = inputs['attention_mask'].unsqueeze(-1)
    masked = mask*outputs.last_hidden_state
    pooled = masked.sum(dim=1)
    sent_emb = pooled/inputs['attention_mask'].sum(dim=1,keepdim=True)
    return sent_emb

top1_correct = 0
top5_correct = 0
total = len(data)

for i in range(0,total,32):
    batch = data['text'][i:i+32]
    true_labels = data['label_text'][i:i+32]
    batch_embeds = get_embed(batch)
    similarities = F.cosine_similarity(batch_embeds.unsqueeze(1),
                                       embeds.unsqueeze(0),
                                       dim=2)
    values, indices = torch.topk(similarities,k=5,dim=1)
    for j in range(len(batch)):
        retrieved_labels = [labels[index] for index in indices[j]]
        if retrieved_labels[0] == true_labels[j]:
            top1_correct += 1
        if true_labels[j] in retrieved_labels:
            top5_correct += 1
print(f'Top-1 Accuracy: {top1_correct/total:.4f}')
print(f'Top-5 Accuracy: {top5_correct/total:.4f}')