from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch



dataset = load_dataset('mteb/banking77')
data = dataset['train']
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
texts = data['text']
all_embeds = []

for i in range(0,len(texts),32):
    batch = texts[i:i+32]
    embeds = get_embed(batch)
    all_embeds.append(embeds)
all_embeds = torch.cat(all_embeds)
torch.save({
    'embeddings': all_embeds,
    'texts': list(texts),
    'labels': list(data['label_text'])
},'semantic_index.pt')