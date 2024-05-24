# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
#
# models = "defog/sqlcoder-7b-2"
# tokenizer = AutoTokenizer.from_pretrained(models)
# model = AutoModelForCausalLM.from_pretrained(models)
#
# instructions = """напишите SQL запрос, чтобы выбрать все данные из таблицы table_1"""
#
# inputs = tokenizer(instructions, return_tensors="pt")
#
# pad_token_id = tokenizer.eos_token_id
#
# output = model.generate(**inputs, max_length=2048, num_return_sequences=1, pad_token_id=pad_token_id)
#
# generate_text = tokenizer.decode(output[0], skip_special_tokens=True)
#
# print(generate_text)

from openai import OpenAI

# Point to the local server
# client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
#
# history = [
#     {"role": "system",
#      "content": """create table user(
# id_user int(20) primary key;
# name str(20);
# sername str(20);
# age int(20;))
#
# create table work(
# id_job int(20) primary key;
# id_user int(20);
# description str(20);)
#
# ---- connect table user t join table work v: t.id_user=v.id_user.  """},
#     {"role": "user",
#      "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise."},
# ]
#
# while True:
#     completion = client.chat.completions.create(
#         model="RichardErkhov/defog_-_sqlcoder-7b-2-gguf",
#         messages=history,
#         temperature=0.7,
#         stream=True,
#     )
#
#     new_message = {"role": "assistant", "content": ""}
#
#     for chunk in completion:
#         if chunk.choices[0].delta.content:
#             print(chunk.choices[0].delta.content, end="", flush=True)
#             new_message["content"] += chunk.choices[0].delta.content
#
#     history.append(new_message)
#
#     # Uncomment to see chat history
#     # import json
#     # gray_color = "\033[90m"
#     # reset_color = "\033[0m"
#     # print(f"{gray_color}\n{'-'*20} History dump {'-'*20}\n")
#     # print(json.dumps(history, indent=2))
#     # print(f"\n{'-'*55}\n{reset_color}")
#
#     print()
#     history.append({"role": "user", "content": input("> ")})
# Example: reuse your existing OpenAI setup
from openai import OpenAI
import json
qyest = input(">>>>")
# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

completion = client.chat.completions.create(
  model="defog/sqlcoder-7b-2",
  messages=[
    {"role": "system", "content": """create table user(
 id_user int(20) primary key;
 name str(20);
 sername str(20);
 age int(20;))

 create table work(
 id_job int(20) primary key; 
 id_user int(20);
 name_company str(20);)

 ---- connect table user t join table work v: t.id_user=v.id_user.  """},
    {"role": "user", "content": f'+{qyest}+' }
  ],
  temperature=0.7,
)
answer_sql = str(completion.choices[0].message)
temp = answer_sql.replace("', role='assistant', function_call=None, tool_calls=None)", "")
answer_sql = temp.replace("ChatCompletionMessage(content='", "")
print(answer_sql)
