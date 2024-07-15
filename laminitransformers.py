from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import psutil
from psutil._common import bytes2human
import sys
from time import sleep
import datetime
"""
import psutil
from psutil._common import bytes2human

mem_usage = psutil.virtual_memory()
total_in_human_format = bytes2human(mem_usage[0])
print(total_in_human_format)
"""
mem_usage = psutil.virtual_memory()
total_in_human_format = bytes2human(mem_usage[3])
print(f'Acutal used memory: {total_in_human_format}')
LaMini = './LaMini-Flan-T5-248M/'
print('Loading the model LaMini-Flan-T5-248M...')
tokenizer = AutoTokenizer.from_pretrained(LaMini)
model = AutoModelForSeq2SeqLM.from_pretrained(LaMini,
                                            device_map='cpu',
                                            torch_dtype=torch.float32)
print('Creating the pipeline...')
llm = pipeline('text2text-generation', 
                model = model,
                tokenizer = tokenizer,
                max_length = 512, 
                do_sample=True,
                temperature=0.35,
                top_p=0.8,
                repetition_penalty = 1.3,
                top_k = 4,
                penalty_alpha = 0.6
                )
mem_usage = psutil.virtual_memory()
total_in_human_format = bytes2human(mem_usage[3])
print(f'RAM in use: {total_in_human_format}')
while True:      
    userinput = ""
    print("\033[1;30m")  #dark grey
    print("Enter your text (end input with Ctrl+D on Unix or Ctrl+Z on Windows) - type quit! to exit the chatroom:")
    print("\033[91;1m")  #red
    lines = sys.stdin.readlines()
    for line in lines:
        userinput += line + "\n"
    if "quit!" in lines[0].lower():
        print("\033[0mBYE BYE!")
        break
    print("\033[92;1m")

    full_response = ""
    print("\033[1;30m")  #dark grey
    start = datetime.datetime.now()
    results = llm(userinput)
    delta = datetime.datetime.now() - start
    mem_usage = psutil.virtual_memory()
    total_in_human_format = bytes2human(mem_usage[3])
    print(f'RAM in use: {total_in_human_format}')
    print(f'Generation time: {delta}')
    print("\033[92;1m")
    for chunk in results[0]['generated_text']:
                print(chunk, end="", flush=True)
                sleep(0.012) 

