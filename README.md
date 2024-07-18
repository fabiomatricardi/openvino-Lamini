# openvino-Lamini
test openvino


#### References
- https://huggingface.co/docs/optimum/main/intel/openvino/export
- https://github.com/openvinotoolkit/openvino
- https://huggingface.co/docs/optimum/main/intel/openvino/inference
- https://huggingface.co/docs/optimum/main/intel/openvino/reference
- https://github.com/intel-analytics/ipex-llm
- https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/Quickstart/install_windows_gpu.md


### Create `venv` and install dependencies
```
➜ python -m venv venv
➜ venv\Scripts\activate
pip install --upgrade --upgrade-strategy eager "optimum[openvino]"
pip install accelerate
pip install streamlit==1.36.0 tiktoken

▶ optimum-cli export openvino --model .\model248M\ --task text2text-generation-with-past --weight-format int8 lamini248M_ov/
```

transformation took 5 minutes

Final model size 60% smaller


### Inference
```
from optimum.intel import OVModelForSeq2SeqLM
from transformers import AutoTokenizer, pipeline
import sys
from time import sleep

model_id = "./lamini248M_ov"

model = OVModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline("text2text-generation", 
                    model=model, 
                    tokenizer=tokenizer,
                    max_length = 512, 
                    do_sample=True,
                    temperature=0.35,
                    top_p=0.8,
                    repetition_penalty = 1.3)
#                    top_k = 4,
#                    penalty_alpha = 0.6
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
    results = pipe(userinput)
    for chunk in results[0]['generated_text']:
                print(chunk, end="", flush=True)
                sleep(0.012)                        
   
```

