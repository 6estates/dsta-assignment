import argparse
import json
import pathlib
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def process_arguments():
    parser = argparse.ArgumentParser(description="llm inference")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Name or path of the model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        required=False,
    )
    parser.add_argument("--sid", type=int, help="ID of the sample(0 to 1662)", default=random.randint(0, 100), required=False)
    args = parser.parse_args()
    return args.model_name_or_path, args.sid

def main():
    model_name_or_path, sample_id = process_arguments()

    # check arguments
    print("Model Name or Path:", model_name_or_path)
    print("Sample ID:", sample_id)

    infer_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if tokenizer.pad_token_id is None:
        if tokenizer.bos_token == '<s>':
            # set <unk> for llama2
            tokenizer.pad_token_id = 0
        else:
            # set <|reserved_special_token_250|> for llama3
            tokenizer.pad_token_id = 128255

    sample = json.loads((pathlib.Path(__file__).parent / "data/test.json").read_text())[sample_id]

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a financial assistant. Always provide accurate and reliable information to the best of your abilities"},
            {"role": "user", "content": sample["user_prompt"]}
        ],
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    outputs = infer_model.generate(
        inputs, do_sample=False, num_beams=1, temperature=None, top_p=None, pad_token_id=tokenizer.pad_token_id
    )
    completion_only_outputs = outputs[0][inputs.shape[1]:]
    if completion_only_outputs[-1] == tokenizer.eos_token_id:
        completion_only_outputs = completion_only_outputs[:-1]

    print(f">>>>>>>>>>>>>>>>>>>>>>>Prompt:\n{sample['user_prompt']}")
    print(f"\n\n<<<<<<<<<<<<<<<<<<<<<<<Response:\n{tokenizer.decode(completion_only_outputs)}")


if __name__ == '__main__':
    main()
