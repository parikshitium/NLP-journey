import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from transformers import TrainingArguments, Trainer
import numpy as np



device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = load_dataset('stanfordnlp/imdb')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
).to(device)
def tokenize(batch):
    return tokenizer(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=256
    )
tokenized_dtset = dataset.map(
    tokenize,
    batched=True
)
train_dtset = tokenized_dtset['train'].shuffle(seed=42).select(range(5000))
test_dtset = tokenized_dtset['test'].shuffle(seed=42).select(range(1000))
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions,axis=1)
    return {
        'accuracy': (predictions == labels).mean()
    }

training_arg = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy='epoch',
    save_strategy='no',
    logging_steps=100,
    report_to='none'
)
trainer = Trainer(
    model=model,
    args=training_arg,
    train_dataset=train_dtset,
    eval_dataset=test_dtset,
    compute_metrics=compute_metrics
)
trainer.train()
trainer.save_model('./sentiment_model')
results = trainer.evaluate()
print(results)
