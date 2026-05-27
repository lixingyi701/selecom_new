import os
from transformers import PrinterCallback
import json


# Prompt template for selector of Stage 1 with <ENCODE> tokens
def get_encode_prompt(model_name, questions, documents, answers, num_pos):
    soft_prompts = []

    if "Qwen3" in model_name:
        for question, document_list, answer in zip(questions, documents, answers):
            for document in document_list:
                instruction = f'### Document\n{document}\n\n### Question\n{question}\n\n### Instruction\nExtract the key information from the document that is helpful to answer the question.'
                output_placeholder = '<ENCODE>' * num_pos
                prompt = f'<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output_placeholder}<|im_end|>'
                soft_prompts.append(prompt)

    return soft_prompts


# Prompt template for generator of Stage 1 with <SOFT_PROMPT> tokens
def get_select_prompt(model_name, questions, answers, num_pos):
    soft_prompts = []

    if 'Qwen' in model_name:
        for question, answer in zip(questions, answers):
            input_placeholder = '<SOFT_PROMPT_START>' + '<SOFT_PROMPT>' * num_pos + '<SOFT_PROMPT_END>'
            instruction = f'### Reference\n{input_placeholder}\n\n### Question\n{question}\n\n### Instruction\nAnswer the question according to the reference provided above.\n\n### Restriction\n'
            instruction += "1. You must use English.\n"
            instruction += "2. You must DIRECTLY provide the answer in this STRICT format:\n"
            instruction += "<answer>YOUR ANSWER</answer>\n"
            instruction += "3. You must not generate any other text."
            prompt = f'<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n'
            prompt += f'<answer>{answer.strip(".")}</answer><|im_end|>'
            soft_prompts.append(prompt)
    elif 'Mistral' in model_name:
        for question, answer in zip(questions, answers):
            input_placeholder = '<SOFT_PROMPT_START>' + '<SOFT_PROMPT>' * num_pos + '<SOFT_PROMPT_END>'
            instruction = f'### Reference\n{input_placeholder}\n\n### Question\n{question}\n\n### Instruction\nAnswer the question according to the reference provided above.\n\n### Restriction\n'
            instruction += "1. You must use English.\n"
            instruction += "2. You must DIRECTLY provide the answer in this STRICT format:\n"
            instruction += "<answer>YOUR ANSWER</answer>\n"
            instruction += "3. You must not generate any other text."
            prompt = f'[INST]\n{instruction} [/INST]\n'
            prompt += f'<answer>{answer.strip(".")}</answer></s>'
            soft_prompts.append(prompt)

    return soft_prompts


# Prompt template for generator of Stage 2 with <SOFT_PROMPT> tokens
def get_qa_prompt(model_name, questions, documents, answers, num_pos, test=False):
    soft_prompts = []
    
    if 'Qwen' in model_name:
        for question, document_list, answer in zip(questions, documents, answers):
            input_placeholder = '<SOFT_PROMPT_START>' + '<SOFT_PROMPT>' * num_pos + '<SOFT_PROMPT_END>'
            if len(document_list) == 1:
                references = f'### Reference\n{input_placeholder}'
                instruction = f'{references}\n\n### Question\n{question}\n\n### Instruction\nAnswer the question using the reference provided above.\n\n'
            else:
                references = '\n\n'.join([f'### Reference {i + 1}\n{input_placeholder}' for i in range(len(document_list))])
                instruction = f'{references}\n\n### Question\n{question}\n\n### Instruction\nAnswer the question using the references provided above.\n\n'
            instruction += '### Restriction\n'
            instruction += "1. You must use English.\n"
            instruction += "2. You must DIRECTLY provide the answer in this STRICT format:\n"
            instruction += "<answer>YOUR ANSWER</answer>\n"
            instruction += "3. You must not generate any other text."
            prompt = f'<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n'
            if not test:
                prompt += f'<answer>{answer.strip(".")}</answer><|im_end|>'
            soft_prompts.append(prompt)
    elif 'Mistral' in model_name:
        for question, document_list, answer in zip(questions, documents, answers):
            input_placeholder = '<SOFT_PROMPT_START>' + '<SOFT_PROMPT>' * num_pos + '<SOFT_PROMPT_END>'
            if len(document_list) == 1:
                references = f'### Reference\n{input_placeholder}'
                instruction = f'{references}\n\n### Question\n{question}\n\n### Instruction\nAnswer the question using the reference provided above.\n\n'
            else:
                references = '\n\n'.join([f'### Reference {i + 1}\n{input_placeholder}' for i in range(len(document_list))])
                instruction = f'{references}\n\n### Question\n{question}\n\n### Instruction\nAnswer the question using the references provided above.\n\n'
            instruction += '### Restriction\n'
            instruction += "1. You must use English.\n"
            instruction += "2. You must DIRECTLY provide the answer in this STRICT format:\n"
            instruction += "<answer>YOUR ANSWER</answer>\n"
            instruction += "3. You must not generate any other text."
            prompt = f'[INST]\n{instruction} [/INST]\n'
            if not test:
                prompt += f'<answer>{answer.strip(".")}</answer></s>'
            soft_prompts.append(prompt)

    return soft_prompts


# Utility to compute and print the number of trainable parameters
def compute_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f'All parameters: {all_params}')
    print(f'Trainable parameters: {trainable_params}')


# Custom callback to log training metrics to a file
class FilePrinterCallback(PrinterCallback):
    def __init__(self, output_file: str):
        super().__init__()
        self.output_file = output_file
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        open(self.output_file, "w").close()

    def on_log(self, args, state, control, logs=None, **kwargs):
        super().on_log(args, state, control, logs=logs, **kwargs)
        
        record = {"step": state.global_step, **(logs or {})}
        with open(self.output_file, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
