import torch
from sentence_transformers import SentenceTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
index = torch.load('classifier.pt',
                   map_location=device,
                   weights_only=False)
model = index['model']
labels = index['labels']
name = 'BAAI/bge-base-en-v1.5'
embed = SentenceTransformer(name).to(device)
query = input('Query: ')
qry_embedding = embed.encode([query],
                             normalize_embeddings=True)
qry_pred = model.predict(qry_embedding)
print(f'Intent: {labels[int(qry_pred[0])]}')