from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch
# from sokets import _send_text


device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = "defog/sqlcoder-7b-2"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model)


pipe = pipeline("text-generation", model=model, model_kwargs = {"torch_dtype" : torch.bfloat16}, device_map = "auto")
answer = "напиши SQL апрос для выгрузки всех данных из таблицы table_1"

rezult = pipe(answer)
print(rezult)
