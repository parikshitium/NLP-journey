import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

from transformers import pipeline
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
ner = pipeline('ner',
               model='dbmdz/bert-large-cased-finetuned-conll03-english',
               aggregation_strategy='simple')

def entities(text):
    result = ner(text)
    entities = []

    for entity in result:
        entities.append({
            'text': entity['word'],
            'label': entity['entity_group'],
            'confidence': round(float(entity['score']),4)
        })
    return entities

if __name__ == '__main__':
    text = input('Text: ')
    print(entities(text))