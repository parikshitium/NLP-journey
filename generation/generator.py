from transformers import AutoTokenizer, AutoModelForCausalLM

name = 'openai-community/gpt2'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

prompt = input('Prompt: ')
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs,
                         max_new_tokens=100,
                         do_sample=True,
                         repetition_penalty=1.2,
                         temperature=0.7,
                         top_p=0.9)
print(tokenizer.decode(outputs[0],skip_special_tokens=True))