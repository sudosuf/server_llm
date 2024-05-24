import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("defog/sqlcoder-7b-2")
model = AutoModelForCausalLM.from_pretrained("defog/sqlcoder-7b-2", low_cpu_mem_usage=True, use_safetensors=True)

pipe = pipeline("text-generation", model="defog/sqlcoder-7b-2", max_length=256, device_map="auto")

while True:
  question = input(">>>>>>")
  otputs = pipe(str(question))
  stringin=str(otputs)
  print(stringin.replace(question, ""))
