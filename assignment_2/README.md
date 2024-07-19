Assignment 2
====================

Let's finetune a llm!

Please review all the instructions and address the questions presented at each stage.

### Requirements
Environment is ready to use, and model files have been downloaded.
You can activate the shared environment using the following command:

```bash
conda activate assignment
```
You can find the model files at the following locations:

```bash
/public/Llama-2-7b-chat-hf
/public/Meta-Llama-3-8B-Instruct
```

<details>
<summary>If you are interested, you can expand here and try installing the environment yourself by following the steps below:</summary>

<br>

 Step 1: Install CUDA11.8 

```bash
wget https://raw.githubusercontent.com/TimDettmers/bitsandbytes/main/install_cuda.sh
bash install_cuda.sh 118 $HOME/cuda

# add to ~/.bashrc
echo 'export LD_LIBRARY_PATH=$HOME/cuda/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PATH=$HOME/cuda/cuda-11.8/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# check if it is 11.8
nvcc --version
```

Step 2: To create an environment with [MiniConda](https://docs.conda.io/en/latest/miniconda.html) and activate it.

```bash
conda create -n assignment2 python=3.9
conda activate assignment2
pip install -r requirements.txt
```

Step 3: Downloads llama2/llama3

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# Downloads llama2
# When prompted for a password, use an access token with write permissions.
# Generate one from your settings: https://huggingface.co/settings/tokens
# Make sure you have access to llama2 model: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

# Downloads llama3
# Enter the token as password as well
# Make sure you have access to llama3 model: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
```
</details>

### Dataset

The training data can be accessed from the folder `data`. It is a large-scale QA dataset for research involving complex tabular and textual data with numerical reasoning. It includes overall 16,552 questions on 2,757 contexts from financial reports, featuring diverse answer forms and detailed annotations.

### Train

Parameter-Efficient finetuning Llama-3-8B-Instruct on 1 X A100 40GB GPU

```bash
# Update the OUTPUT_DIR and MODEL_PATH
# MODEL_PATH supports "/public/Meta-Llama-3-8B-Instruct" and "/public/Llama-2-7b-chat-hf"
CUDA_VISIBLE_DEVICES=0 bash finetune_lora.sh
```

<details>
<summary><strong>Question1</strong></summary>

<br>

>**Question1.1:**
>Train the model using LoRA instruction above and monitor the change in loss during training. Does it behave as expected?  Note: Since the entire training process may take a long time, press `Ctrl+c` to stop monitoring once you have gathered enough information.

>**Question1.2:**
>You can see that the default values for `per_device_train_batch_size` and `per_device_eval_batch_size` are currently set to 1, but these can be adjusted for improved performance. Increasing the batch size can accelerate training, although it may lead to out-of-memory errors. Determine the optimal batch size for LoRA training by observing the time taken to train and GPU memory usage with the `nvidia-smi` command in a separate terminal window. Does GPU memory usage change as expected with varying batch sizes? Again, press 'Ctrl+c' to stop monitoring once you have gathered enough information.
</details>

<details>
<summary><strong>Question2</strong></summary>

 <br>
 
>Let’s further delve into the exploration of the LoRA training.
Thoroughly review the hyperparameters in the `finetune_lora.sh` script, aside from batch sizes. Consider which other hyperparameters might decrease GPU memory consumption. You might experiment by removing one to observe its impact on GPU memory usage.

>Refer to the parameter descriptions in the following documentation for guidance:
>- [Transformers Trainer Arguments](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments)
>- [Model Config](https://huggingface.co/docs/trl/en/sft_trainer#trl.ModelConfig)

<br>

>Additionally, Investigate other potential hyperparameters (in the documentation) for the LoRA training that could benefit for the GPU memory usage.
</details>

Full-Parameter finetuning Llama-3-8B-Instruct on 4 X A100 40GB GPU

```bash
# Update the OUTPUT_DIR and MODEL_PATH
# MODEL_PATH supports "/public/Meta-Llama-3-8B-Instruct" and "/public/Llama-2-7b-chat-hf"
bash finetune_fft.sh
```

<details>
<summary><strong>Question3</strong></summary>

<br>

>Following the guidelines from Question 1, repeat those experiments for FFT training. Observe the time taken to train and GPU memory usage. Then, with the optimal training setting, compare LoRA and FFT in terms of these two metrics. Analyze and explain the reasons behind any differences observed between the two training methods.
</details>

### Inference

Inference is the process of using a trained machine learning model to make predictions or decisions based on new, unseen data.

The completed fine-tuned Llama3 models for LoRA and FFT are already available at `/public/Meta-Llama-3-8B-Instruct-Lora` and `/public/Meta-Llama-3-8B-Instruct-FFT`.

<details>
<summary>If you're interested in the complete fine-tuning process, click here for additional steps needed before inference.</summary>

<br>
The model fine-tuned by LoRA will be generated immediately after its training completes. However, for FFT, an additional step is required to finalize the model, which involves running the zero_to_fp32.py script.

Run the `zero_to_fp32.py` script with the following usage:

```bash
# usage: zero_to_fp32.py [-h] [-t TAG] [-d] checkpoint_dir output_file

# positional arguments:
#  checkpoint_dir      path to the desired checkpoint folder, e.g., path/checkpoint-12
#  output_file         path to the pytorch fp32 state_dict output file (e.g. path/checkpoint-12/pytorch_model.bin)

#optional arguments:
#  -h, --help          show this help message and exit
#  -t TAG, -- tag TAG  checkpoint tag used as a unique identifier for checkpoint. e.g., global_step1
#  -d, --debug         enable debug

# update the checkpoint_dir and output_file accordingly
python zero_to_fp32.py path/checkpoint-12  path/checkpoint-12/pytorch_model.bin
```
</details>

<br>

To move to the inference stage, execute the inference.py script using the following command:

```bash
# usage: inference.py [-h] [--model_name_or_path MODEL_NAME_OR_PATH] [--sid SID]

# llm inference

# optional arguments:
#  -h, --help            show this help message and exit
#  --model_name_or_path MODEL_NAME_OR_PATH
#                        Name or path of the model
#  --sid SID             ID of the sample(0 to 1643)
CUDA_VISIBLE_DEVICES=0 python inference.py --sid 100
```
<details>
<summary><strong>Question4</strong></summary>

<br>

>Run inference on the original Llama2 and Llama3 models without fine-tuning. Specifying the model name of `/public/Meta-Llama-3-8B-Instruct` or `/public/Llama-2-7b-chat-hf` in the inference instruction: `CUDA_VISIBLE_DEVICES=0 python inference.py [--model_name_or_path MODEL_NAME_OR_PATH] --sid 100`. Evaluate the outputs from both Llama2 and Llama3 by comparing the model responses to the sample answers. Vary the sid 100 parameter to test different IDs like 100, 500, 900, among others, to review their performance. Based on your analysis, which model do you think performs better?
</details>

<details>
<summary><strong>Question5</strong></summary>

<br>

>Run inference on the models that have been fine-tuned and compare the difference between model answer and sample answer. Again, you can test different sample IDs by adjusting the `sid 100` parameter to observe various results. Which model do you think performs the best: the original, LoRA fine-tuned, or FFT fine-tuned?
</details>
