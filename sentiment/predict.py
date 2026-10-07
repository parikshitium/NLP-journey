from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(
    './sentiment_model'
)
review = input('Enter a review: ')
inputs = tokenizer(
    review,
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors='pt'
)
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits,dim=1).item()
    prob = torch.softmax(outputs.logits,dim=1)
    confidence = prob[0][prediction].item()
label = 'Positive' if prediction == 1 else 'Negative'
print(f'Prediction: {label}')
print(f'Confidence: {confidence:.2%}')