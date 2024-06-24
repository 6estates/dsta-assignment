Assignment 2
====================

Let's finetune a llm!

### Requirements

<details>
<summary> Step 1: Install CUDA11.8 </summary>

```bash
wget https://raw.githubusercontent.com/TimDettmers/bitsandbytes/main/install_cuda.sh
# replace /xxx/cuda to the correct path
bash install_cuda.sh 118 /xxx/cuda

# replace /xxx/cuda to the correct path
# add to ~/.bashrc
echo 'export LD_LIBRARY_PATH=/xxx/cuda/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PATH=/xxx/cuda/cuda-11.8/bin:$PATH' >> ~/.bashrc

# check if it is 11.8
nvcc --version
```

</details>

<br>

Step 2: To create an environment with [MiniConda](https://docs.conda.io/en/latest/miniconda.html) and activate it.

```bash
conda create -n assignment python=3.9
conda activate assignment
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirement.txt
```

<details>
<summary> Step 3: Downloads llama2/llama3 </summary>

```bash
export TOKEN='xxxx'
# downloads llama2
python -c "import os; from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Llama-2-7b-chat-hf', token=os.environ['TOKEN'])"
# downloads llama3
python -c "import os; from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', token=os.environ['TOKEN'])"
```
</details>

### Dataset

The training data can be accessed from the folder `data`.

### Train

Parameter-Efficient finetuning Llama-3-8B-Instruct on 1 X A100 40GB GPU

```bash
# Update the OUTPUT_DIR and MODEL_PATH
# MODEL_PATH supports "meta-llama/Meta-Llama-3-8B-Instruct" and "meta-llama/Llama-2-7b-chat-hf"
CUDA_VISIBLE_DEVICES=0 bash finetune_lora.sh
```

Full-Parameter finetuning Llama-3-8B-Instruct on 8 X A100 40GB GPU

```bash
# Update the OUTPUT_DIR and MODEL_PATH
# MODEL_PATH supports "meta-llama/Meta-Llama-3-8B-Instruct" and "meta-llama/Llama-2-7b-chat-hf"
bash finetune_fft.sh
```


### Inference

```bash
# usage: inference.py [-h] [--model_name_or_path MODEL_NAME_OR_PATH] [--sid SID]

# llm inference

# optional arguments:
#  -h, --help            show this help message and exit
#  --model_name_or_path MODEL_NAME_OR_PATH
#                        Name or path of the model
#  --sid SID             ID of the sample(0 to 1662)
CUDA_VISIBLE_DEVICES=0 python inference.py --sid 100
```
