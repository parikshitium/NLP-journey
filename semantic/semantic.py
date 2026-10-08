import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F



index = torch.load('semantic_index.pt')
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
    with torch.no_grad():
        outputs = model(**inputs)
    mask = inputs['attention_mask'].unsqueeze(-1)
    masked = mask*outputs.last_hidden_state
    pooled = masked.sum(dim=1)
    sent_emb = pooled/inputs['attention_mask'].sum(dim=1,keepdim=True)
    return sent_emb 
embeds = index['embeddings']
texts = index['texts']
labels = index['labels']

query = input('Query: ')
qry_embed = get_embed([query])
similarity = F.cosine_similarity(qry_embed,embeds)

vals, indices = torch.topk(similarity,k=5)
for val, index in zip(vals,indices):
    print(f'{val:.4f} | {texts[index]} | {labels[index]}')