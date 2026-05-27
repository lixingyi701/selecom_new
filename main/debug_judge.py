import sys
sys.path.append('..')
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from util.util import load_jsonl

model_name = "../baselineModel/Mistral-7B-Instruct-v0.2"
result_path = "../results/eval_results/nq_top1_results.jsonl"

data = load_jsonl(result_path)[:20]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
llm = LLM(model=model_name, tensor_parallel_size=4, gpu_memory_utilization=0.9,
          max_model_len=4096, trust_remote_code=True, dtype=torch.bfloat16)
sampling_params = SamplingParams(temperature=0.2, max_tokens=10)

system_prompt = ("You are an evaluation assistant. You will be given a question, a candidate answer, "
                 "and a reference answer. Judge whether the candidate answer correctly addresses the "
                 "question compared to the reference. Respond **ONLY** with a numeric score between "
                 "0 (completely wrong) and 1 (perfectly correct).")

prompts = []
for item in data:
    gt = item['groundtruth'][0] if isinstance(item['groundtruth'], list) else item['groundtruth']
    user_prompt = f"Question: {item['question']}\nCandidate Answer: {item['output']}\nReference Answer: {gt}\nScore:"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

outputs = llm.generate(prompts, sampling_params)

for i, (item, output) in enumerate(zip(data, outputs)):
    gt = item['groundtruth'][0] if isinstance(item['groundtruth'], list) else item['groundtruth']
    text = output.outputs[0].text
    print(f"[{i}] Q: {item['question'][:60]}")
    print(f"     A: {item['output'][:60]}")
    print(f"     GT: {gt[:60]}")
    print(f"     RAW OUTPUT: {repr(text)}")
    print()
