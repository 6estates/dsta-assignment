Assignment 4
====================

Let's use the following engines to infer and serve a llm model !

#### Inference engines
LLM model can infer and serve in various engines and formats.
The following list the model inference services introduced in this assigment.

- Fastchat command line interface
- Fastchat api server
- Fastchat web-ui
- Ollama command line interface
- Vllm sdk
- Vllm api server
- TensorRT-LLM sdk
- TensorRT-LLM Triton server
- Onnxruntime-gpu

#### Environment
In this assignment ,  we use Nvidia NGC docker as llm inference runtime environment which has provided base libraries for llm serving.

All scripts in this assignment can be executed successfully in the Nvidia base image version: nvcr.io/nvidia/tritonserver:24.06-trtllm-python-py3

- GPU: Nvidia A100
- Cuda version: 12.5
- Runtime: Nvidia NGC containers

#### Model
- Meta-Llama-3-8B-Instruct
- Meta-Llama-3-70B-Instruct

## Fastchat
An open platform for training, serving, and evaluating large language model based chatbots.
#### Fastchat command line interface

```bash
sudo docker run -it --gpus all  \
          --shm-size 25g \
          --net host     \
          -v /public/:/llm/  \
    nvcr.io/nvidia/tritonserver:24.06-trtllm-python-py3 /bin/bash

pip install fschat==0.2.36

python -m fastchat.serve.cli --model-path /llm/Meta-Llama-3-70B-Instruct/ --num-gpus 4
```
Then you can chat with LLama3 in this cli(enter Ctrl C to interrupt conversation):
```bash
>>> Human: Who won the world cup in 2006?
Assistant: The 2006 FIFA World Cup was won by Italy! They defeated France 5-3 in the final penalty shootout, after the match ended 1-1 after extra time, on July 9, 2006, at the Olympiastadion in Berlin, Germany. It was Italy's fourth World Cup title.
```

#### Fastchat api server and web-ui
To serve using the restful api and web UI, you need three main components: web servers that interface with users, model workers that host one or more models, and a controller to coordinate the webserver and model workers. Here are the commands to follow in your terminal:
```bash
#start runtime container
sudo docker run -it --gpus all  \
          --shm-size 25g \
          --net host     \
          -v /public/:/llm/  \
    nvcr.io/nvidia/tritonserver:24.06-trtllm-python-py3 /bin/bash

pip install fschat==0.2.36

#controller
python -m fastchat.serve.controller --host 0.0.0.0 --port 10002 &

#worker
python -u -m fastchat.serve.model_worker --model-name 'llama70b' \
    --model-path /llm/Meta-Llama-3-70B-Instruct/  \
    --controller http://localhost:10002 \
    --port 31000 \
    --num-gpus 4 \
    --worker http://localhost:31000 &

#api server
python  -m fastchat.serve.openai_api_server --controller-address http://localhost:10002 --host localhost --port 8000 &

#web-ui
pip install gradio plotly

python -m fastchat.serve.gradio_web_server_multi --controller http://localhost:10002 &
```
Then you can post a http request to chat with Llama3 :
```bash   
curl http://localhost:8000/v1/chat/completions \
 -H "Content-Type: application/json" \
 -d '{
   "model": "llama70b",
   "messages": [{"role": "user", "content": "Human: Who won the world cup in 2006?"}],
   "max_tokens":20
 }'

```
Open the web browser and enter the url http://localhost:7860/ to chat in the web-ui.




## Ollama

Ollama is an open-source project that serves as a powerful and user-friendly platform for running LLMs on your local machine. It acts as a bridge between the complexities of LLM technology and the desire for an accessible and customizable AI experience.

#### Ollama command line interface
```bash
#start runtime container
sudo docker run -it --gpus all  \
                --shm-size 25g \
                --net host     \
                -v /public/:/llm/  \
    nvcr.io/nvidia/tritonserver:24.06-trtllm-python-py3 /bin/bash

#install
curl -fsSL https://ollama.com/install.sh | sh

#start serve
ollama serve &

#download model
ollama pull llama3:8b

#run cli
ollama run llama3

>>> who won the world cup in 2006?
The winner of the FIFA World Cup 2006 was Italy. They defeated France 5-3 in a penalty shootout after the match ended 1-1 after extra 
time, winning their fourth World Cup title. The final was held on July 9, 2006 at the Olympiastadion in Berlin, Germany.[GIN] 2024/07/19 - 02:37:57 | 200 |  670.499487ms |       127.0.0.1 | POST     "/api/chat"


>>> Send a message (/? for help)
>>> 
Use Ctrl + d or /bye to exit.
```


## Vllm
vLLM is a fast and easy-to-use library for LLM inference and serving.
#### Vllm sdk
```bash
#start runtime container
sudo docker run -it --gpus all  \
                --shm-size 25g \
                --net host     \
                -v /public/:/llm/  \
    nvcr.io/nvidia/tritonserver:24.06-trtllm-python-py3 /bin/bash

#install
pip install vllm==0.5.0.post1

```
Infer by python sdk:
```python
root@dsta-02:/opt/tritonserver# python
Python 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>from vllm import LLM
>>>from vllm import SamplingParams
#load llama70B model, may take ten minutes
>>>llm = LLM("/llm/Meta-Llama-3-70B-Instruct/", tensor_parallel_size=2)
>>>from transformers import AutoTokenizer
>>>tokenizer = AutoTokenizer.from_pretrained("/llm/Meta-Llama-3-70B-Instruct/")
>>>messages = [{"role": "user", "content": "Who won the world cup in 2006?"}]
>>>formatted_prompt =  tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
>>>sampling_params=SamplingParams(max_tokens=50)
>>>llm.generate(formatted_prompt, sampling_params=sampling_params)
>>>exit()
```


#### Vllm api server
```bash
#start runtime container
sudo docker run -it --gpus all  \
                --shm-size 25g \
                --net host     \
                -v /public/:/llm/  \
    nvcr.io/nvidia/tritonserver:24.06-trtllm-python-py3 /bin/bash

#install
pip install vllm==0.5.0.post1

#start api server, load llama70B model, may take ten minutes
python3 -m vllm.entrypoints.openai.api_server   \
       --model /llm/Meta-Llama-3-70B-Instruct/  \
       --tensor-parallel-size 2                 \
       --served-model-name "llama70b" &

```
Then you can chat with llama3 by post a http restful request:
```bash
curl http://localhost:8000/v1/chat/completions \
 -H "Content-Type: application/json" \
 -d '{
   "model": "llama70b",
   "messages": [{"role": "user", "content": "Who won the world cup in 2006?"}],
   "max_tokens":100
 }'

```
You can chat with llama3 by Openai client sdk

```python
pip install openai

(base) azureuser@dsta-02:~$ python3
Python 3.12.3 | packaged by Anaconda, Inc. | [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>from openai import OpenAI
>>> 
>>>openai_api_key = "EMPTY" 
>>>openai_api_base = "http://localhost:8000/v1" 
>>> 
>>>client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
>>>messages = [
...     {"role": "user", "content": "Give me a short introduction to large language model."}
... ]
>>>
>>>llm_response = client.chat.completions.create(
...     messages=messages,
...     model="llama70b", 
...     max_tokens=200,
...     temperature=0.1,
...     stream=False   
...)
>>>
>>>print(llm_response.choices[0].message.content)
Here is a short introduction to large language models:

**What is a Large Language Model?**

A large language model is a type of artificial intelligence (AI) designed to process and understand human language. It is a neural network-based model that is trained on a massive dataset of text, allowing it to learn patterns, relationships, and context within language.

**Key Characteristics:**

1. **Scale**: Large language models are trained on enormous datasets, often consisting of billions of parameters and millions of examples.
2. **Deep Learning**: They use deep neural networks to learn complex representations of language.
3. **Contextual Understanding**: They can understand the context in which words are used, enabling them to capture nuances of language.
4. **Generative Capabilities**: They can generate human-like text, including sentences, paragraphs, and even entire articles.

>>>exit() 

```

## TensorRT-LLM
TensorRT-LLM is an easy-to-use Python API to define Large Language Models (LLMs) and build TensorRT engines that contain state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs. TensorRT-LLM contains components to create Python and C++ runtimes that execute those TensorRT engines. It also includes a backend for integration with the NVIDIA Triton Inference Server; a production-quality system to serve LLMs. Models built with TensorRT-LLM can be executed on a wide range of configurations going from a single GPU to multiple nodes with multiple GPUs (using Tensor Parallelism and/or Pipeline Parallelism).


#### TensorRT-LLM sdk
Meta-Llama-3-8B-Instruct on 1 GPU:
```bash  
#start runtime container
sudo docker run -it --gpus 'device=0' \
                --shm-size 25g \
                --net host     \
                -v /public/:/llm/  \
    nvcr.io/nvidia/tritonserver:24.06-trtllm-python-py3 /bin/bash

cd /home
git clone -b v0.10.0 https://github.com/NVIDIA/TensorRT-LLM.git
cd  /home/TensorRT-LLM/examples/llama/
pip install -r requirements.txt

#convert huggingface model to tensorRT format
python convert_checkpoint.py --model_dir /llm/Meta-Llama-3-8B-Instruct/ \
                              --output_dir /llm/trtllm/tllm_checkpoint_1gpu_fp16 \
                              --dtype float16

#prepare TensorRT-LLM engines
trtllm-build --checkpoint_dir /llm/trtllm/tllm_checkpoint_1gpu_fp16 \
              --output_dir /llm/trtllm/trt_engines_fp16_1-gpu \
              --gemm_plugin float16


#chat
python ../run.py --max_output_len=60 --tokenizer_dir /llm/Meta-Llama-3-8B-Instruct  \
    --engine_dir /llm/trtllm/trt_engines_fp16_1-gpu \
    --input_text "In python, write a function for binary searching an element in an integer array."

```

Meta-Llama-3-70B-Instruct on 4 GPUs:
```bash
#start runtime container
sudo docker run -it --gpus all  \
                --shm-size 25g \
                --net host     \
                -v /public/:/llm/  \
    nvcr.io/nvidia/tritonserver:24.06-trtllm-python-py3 /bin/bash

cd /home
git clone -b v0.10.0 https://github.com/NVIDIA/TensorRT-LLM.git
cd  /home/TensorRT-LLM/examples/llama/
pip install -r requirements.txt


#convert huggingface model to tensorRT format
python convert_checkpoint.py --model_dir /llm/Meta-Llama-3-70B-Instruct/ \
                            --output_dir /llm/trtllm/tllm_checkpoint_4gpu_tp4 \
                            --dtype float16 \
                            --tp_size 4

#prepare TensorRT-LLM engines
trtllm-build --checkpoint_dir /llm/trtllm/tllm_checkpoint_4gpu_tp4 \
              --output_dir /llm/trtllm/trt_engines_fp16_4-gpu \
              --gemm_plugin float16 \
              --workers 4

#chat
mpirun -n 4 --allow-run-as-root \
    python ../run.py --max_output_len=160 --tokenizer_dir /llm/Meta-Llama-3-70B-Instruct \
     --engine_dir /llm/trtllm/trt_engines_fp16_4-gpu \
     --input_text "In python, write a function for binary searching an element in an integer array."

```



#### TensorRT-LLM Triton server
Triton Inference Server is an open source inference serving software that streamlines AI inferencing. Triton enables teams to deploy any AI model from multiple deep learning and machine learning frameworks, including TensorRT, TensorFlow, PyTorch, ONNX, OpenVINO, Python, RAPIDS FIL, and more. Triton Inference Server supports inference across cloud, data center, edge and embedded devices on NVIDIA GPUs, x86 and ARM CPU, or AWS Inferentia. Triton Inference Server delivers optimized performance for many query types, including real time, batched, ensembles and audio/video streaming. Triton inference Server is part of NVIDIA AI Enterprise, a software platform that accelerates the data science pipeline and streamlines the development and deployment of production AI.

```bash
#start runtime container
sudo docker run -it --gpus 'device=0'  \
                --shm-size 25g \
                --net host     \
                -v /public/:/llm/  \
    nvcr.io/nvidia/tritonserver:24.06-trtllm-python-py3 /bin/bash

cd /home
git clone  -b v0.10.0  https://github.com/triton-inference-server/tensorrtllm_backend.git

#create the model repository that will be used by the Triton server
cd tensorrtllm_backend
mkdir triton_model_repo

#copy the example models to the model repository
cp -r all_models/inflight_batcher_llm/* triton_model_repo/

#copy the TRT engine built above to triton_model_repo/tensorrt_llm/1/
cp /llm/trtllm/trt_engines_fp16_1-gpu/* triton_model_repo/tensorrt_llm/1/
rm -rf triton_model_repo/tensorrt_llm_bls

#set the tokenizer_dir and engine_dir paths
HF_LLAMA_MODEL=/llm/Meta-Llama-3-8B-Instruct
ENGINE_PATH=triton_model_repo/tensorrt_llm/1

python3 tools/fill_template.py -i triton_model_repo/preprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},tokenizer_type:auto,triton_max_batch_size:4,preprocessing_instance_count:1

python3 tools/fill_template.py -i triton_model_repo/postprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},tokenizer_type:auto,triton_max_batch_size:4,postprocessing_instance_count:1

python3 tools/fill_template.py -i triton_model_repo/ensemble/config.pbtxt triton_max_batch_size:4

python3 tools/fill_template.py -i triton_model_repo/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:4,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0

#launch Server
python3 scripts/launch_triton_server.py --world_size=1 --model_repo=/home/tensorrtllm_backend/triton_model_repo

```
Then you can chat with llama3 by post a http restful request:
```bash
#http request
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": ""}'

#get response
{"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,
"text_output":"What is machine learning? Machine learning is a type of artificial intelligence (AI) that involves training algorithms to make predictions or decisions"}
```
Inference Metrics
```bash
curl -X GET http://localhost:8000/v2/models/ensemble/versions/1/stats
{"model_stats":[{"name":"llama3","version":"1","last_inference":1719233393222,"inference_count":5,"execution_count":5,
"inference_stats":{"success":{"count":5,"ns":19231364485},"fail":{"count":0,"ns":0},"queue":{"count":5,"ns":851917},
"compute_input":{"count":5,"ns":947878},"compute_infer":{"count":5,"ns":19228469863},"compute_output":{"count":5,"ns":787663},
"cache_hit":{"count":0,"ns":0},"cache_miss":{"count":0,"ns":0}},"batch_stats":[{"batch_size":1,"compute_input":{"count":5,"ns":947878},
"compute_infer":{"count":5,"ns":19228469863},"compute_output":{"count":5,"ns":787663}}],"memory_usage":[]}]}
```

## Onnxruntime
ONNX is an open standard that defines a common set of operators and a common file format to represent deep learning models in a wide variety of frameworks, including PyTorch and TensorFlow. When a model is exported to the ONNX format, these operators are used to construct a computational graph (often called an intermediate representation) which represents the flow of data through the neural network.

By exposing a graph with standardized operators and data types, ONNX makes it easy to switch between frameworks. For example, a model trained in PyTorch can be exported to ONNX format and then imported in TensorRT or OpenVINO.

```bash  
#start runtime container
sudo docker run -it --gpus 'device=0'  \
                --shm-size 25g \
                --net host     \
                -v /public/:/llm/  \
    nvcr.io/nvidia/tritonserver:24.06-trtllm-python-py3 /bin/bash

#install
pip install optimum[exporters,onnxruntime]
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
pip install transformers==4.39.3
```



#### convert llama to onnx format
May take half a hour
```bash
optimum-cli export onnx --model  /llm/Meta-Llama-3-8B-Instruct/   /llm/llamaonnx/ --task text-generation --sequence_length 256 --batch_size 1
Framework not specified. Using pt to export the model.
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████| 4/4 [00:05<00:00,  1.28s/it]
The task `text-generation` was manually specified, and past key values will not be reused in the decoding. if needed, please pass `--task text-generation-with-past` to export using the past key values.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Using the export variant default. Available variants are:
    - default: The default ONNX variant.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.

***** Exporting submodel 1/1: LlamaForCausalLM *****
Using framework PyTorch: 2.3.0a0+40ec155e58.nv24.03
Overriding 1 configuration item(s)
    - use_cache -> False
/usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py:1075: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if sequence_length != 1:



Saving external data to one file...


Post-processing the exported models...
Deduplicating shared (tied) weights...

Validating ONNX model /llm/llamaonnx/model.onnx...
    -[✓] ONNX model output names match reference model (logits)
    - Validating ONNX Model output "logits":
        -[✓] (1, 256, 128256) matches (1, 256, 128256)
        -[x] values not close enough, max diff: 4.1484832763671875e-05 (atol: 1e-05)
The ONNX export succeeded with the warning: The maximum absolute difference between the output of the reference model and the ONNX exported model is not within the set tolerance 1e-05:
- logits: max diff = 4.1484832763671875e-05.
 The exported model was saved at: /llm/llamaonnx
```

#### Infer llama by onnxruntime-gpu engine
```python
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

model_id = "/llm/llamaonnx"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = ORTModelForCausalLM.from_pretrained(model_id, provider="CUDAExecutionProvider", use_cache=False, use_merged=False, use_io_binding=False)

inp = tokenizer("role: user, content: Human: who won the world cup in 2006?", return_tensors="pt").to("cuda")

res = model.generate(**inp,max_length=30)

print(tokenizer.batch_decode(res))

```
