# Assignment 3.1

Let's finetune a multimodal LLM! The model we will be working with is MiniCPM-Llama3-V 2.5. It is a high-performance multimodal language model with 8 billion parameters. It offers strong OCR capabilities, surpasses proprietary models, and demonstrates trustworthy behavior with a lower hallucination rate.

This assignment is an excerpt from the [GUICourse-MiniCPM-V-Llama3 repository](https://github.com/qyc-98/MiniCPM-V-LLama3).

## Setup

### Download code
```bash
git clone https://github.com/qyc-98/MiniCPM-V-LLama3.git
```

### Activate Environment
```
conda activate assignment
```

### Download Model
The Model has been downloaded to `/public/MiniCPM-Llama3-V-2_5`.

## Data

The GUICourse data is available at `/public/guicourse`.

GUICourse is a comprehensive set of datasets designed to train visual-based GUI agents using general VLMs. It aims to enhance VLMs' fundamental abilities and GUI knowledge. GUICourse comprises three datasets:

(1) **GUIEnv**: A large-scale dataset for improving VLMs' OCR and grounding abilities, including 10M website page-annotation pairs as pre-training data and 0.7M region-text QA pairs as SFT data.
![example1](https://github.com/qyc-98/MiniCPM-V-LLama3/raw/main/assets/GUIEnv-example.svg)

(2) **GUIAct**: A GUI navigation dataset in website and Android scenarios for enhancing VLMs' knowledge of GUI systems, including 67k single-step and 15k multi-step action instructions.
![example2](https://github.com/qyc-98/MiniCPM-V-LLama3/raw/main/assets/GUIAct-example.svg)

(3) **GUIChat**: A conversational dataset for improving the interaction skills of GUI agents, including 44k single-turn QA pairs and 6k multi-turn dialogues with text-rich images and bounding boxes.
![example3](https://github.com/qyc-98/MiniCPM-V-LLama3/raw/main/assets/GUIChat-example.svg)

### Data Preparation

You can skip the data preparation section and find data at `/public/guicourse`. If you are interested in preparing the data yourself, follow the steps [here](https://github.com/qyc-98/MiniCPM-V-LLama3/blob/main/README.md#data-preparation) to download the data.

The data format is as follows:

```json
[
    {
        "id": "uid_img_C4web50k-0_1426576-split-2_bbox2text_00",
        "image": "/public/guicourse/images/guienv/chunk_0/C4web50k-0_1426576-split-2.png",
        "conversations": [
            {
                "role": "user",
                "content": "<image>\n## Your Task\nIf the input is a string, please give me the element regions including this string. Else if the input is a region, please give me the string (text) in this region.## Input\n<box>(426,937),(481,960)</box>\n## Output\n"
            },
            {
                "role": "assistant",
                "content": "Green Flash"
            }
        ]
    }
]
```

## Training

### Full-parameter Finetuning

Full-parameter finetuning involves updating all parameters of the LLM throughout the training process. Please specify the correct MODEL path, DATA path, and LLM_TYPE in the shell scripts.

```shell
MODEL="/public/MiniCPM-Llama3-V-2_5"
DATA="/public/guicourse/guicourse_training_data.json" # json file
EVAL_DATA="/public/guicourse/guicourse_test_data.json" # json file
LLM_TYPE="llama3"
```

To start your training, run the following script:

```
bash finetune_ds.sh
```

Note: Llama3 uses different chat templates for training and inference. We modified the chat template for training, so remember to restore the original chat template before inference. You can do this by restoring the tokenizer_config.json in the training checkpoint:

```python
{
    ...
    "chat_template": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    ...
}
```

### LoRA Finetuning

LoRA allows lightweight model tuning with only a small subset of parameters being updated. We provide the LoRA implementation based on `peft`. Please specify the correct MODEL path, DATA path, and LLM_TYPE in the shell scripts as described in previous section.

To start your training, run the following script:

```
bash finetune_lora.sh
```

After training, you can load the model using the path to the adapter. We recommend using the absolute path for your pretrained model, as LoRA only saves the adapter, and the absolute path in the adapter configuration JSON file is used to locate the pretrained model.

```
from peft import AutoPeftModel

path_to_adapter="path_to_your_fine_tuned_checkpoint"

model = AutoPeftModel.from_pretrained(
    # path to the output directory
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval()
```

### Followup

For more training details, visit the [GUICourse-MiniCPM-V-Llama3 repository](https://github.com/qyc-98/MiniCPM-V-LLama3/blob/main/README.md#model-fine-tuning-memory-usage-statistics)

For information on inference, visit the [MiniCPM-V_Llama3 official repository](https://github.com/OpenBMB/MiniCPM-V?tab=readme-ov-file#multi-turn-conversation)
