from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer
import numpy as np

dataset = load_dataset('stanfordnlp/imdb')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(
    './sentiment_model'
)
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
test_dtset = tokenized_dtset['test'].shuffle(seed=42).select(range(1000))
def compute_metrics(eval_pred):
    predictions,labels = eval_pred
    predictions = np.argmax(predictions,axis=1)
    return {
        'accuracy':(predictions == labels).mean()
    }
training_args = TrainingArguments(
    output_dir='./eval_results',
    per_device_eval_batch_size=8,
    report_to='none'
)
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dtset,
    compute_metrics=compute_metrics
)
results = trainer.evaluate()
print(results)