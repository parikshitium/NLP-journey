from datasets import load_dataset
import torch
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,classification_report



device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = load_dataset('mteb/banking77')
data = dataset['train']
name = 'BAAI/bge-base-en-v1.5'
embed = SentenceTransformer(name).to(device)
embeddings = embed.encode(data['text'],
                          normalize_embeddings=True,
                          batch_size=32)

y = data['label']
model = LinearSVC(C=3,max_iter=5000)
model.fit(embeddings,y)
test_text,test_labels = dataset['test']['text'],dataset['test']['label']
test_embedding = embed.encode(test_text,
                              normalize_embeddings=True,
                              batch_size=32)
preds = model.predict(test_embedding)
acc = accuracy_score(test_labels,preds)
report = classification_report(test_labels,preds)
print(report)
print(f'Accuracy: {acc*100:.2f}')

mapping_dict = dict(zip(dataset['train']['label'],dataset['train']['label_text']))
label_names = [mapping_dict[i] for i in sorted(mapping_dict.keys())]
saved_index = {
    'model': model,
    'labels': label_names
}
torch.save(saved_index,'classifier.pt')