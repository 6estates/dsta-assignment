Assignment for DSTA LLM Course
======

# Introduction

Welcome to our course, where we provide hands-on training designed to enhance your skills in using Large Language Models (LLMs) and building LLM-related applications. The assignment includes practical code exercises and materials to challenge your LLM proficiency. The training covers the following key topics:
1. Data Preprocessing
2. LLM Training and Inference
3. Retrieval Augmented Generation (RAG) Pipeline
4. Model Serving for Production

# Assignment Overview

The assignment is structured into several sections to provide a systematic learning experience with LLMs. We have created hands-on questions for certain assignments. Once you've attempted them, you can refer to the assignment_hints folder to access the answers and enhance your understanding. Before diving into the details, we first need to set up the conda virtual environment that will be used in this assignment.

## Environment Setup

A default `assignment` environment has been installed on your machine to facilitate your learning experience. Activate it by running the following command:

```bash
conda activate assignment
```

<details><summary>(Optional) click to see how this environment is installed</summary>


  ```bash
  # install cuda 11.8
  wget https://raw.githubusercontent.com/TimDettmers/bitsandbytes/main/install_cuda.sh
  bash install_cuda.sh 118 $HOME/cuda
  echo 'export LD_LIBRARY_PATH=$HOME/cuda/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
  echo 'export PATH=$HOME/cuda/cuda-11.8/bin:$PATH' >> ~/.bashrc
  source ~/.bashrc

  # install tesseract-ocr
  sudo apt-get update -y
  sudo apt-get install tesseract-ocr=4.1.1-2.1build1 -y
  sudo wget https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata
  sudo mv eng.traineddata /usr/share/tesseract-ocr/4.00/tessdata/
  
  # install pillow requirements
  sudo apt install libjpeg-dev zlib1g-dev

  # install git lfs for assignment 3-1
  sudo apt install git-lfs

  # install conda environment
  conda create -n assignment python=3.9 ipykernel
  conda activate assignment
  pip install -r requirements.txt
  ```

</details>

## Assignment 1 (Day 1)

### Assignment 1.1: Data Extraction

Learn how to process documents, such as PDFs and images, to extract text and its corresponding position using PYMUPDF and OCR. The extracted data will be stored in structured JSON files.

### Assignment 1.2: Vector Database Storage

Store the extracted text representations in vector databases like Milvus and Elasticsearch. This allows for efficient text retrieval, which can be utilized by a Retrieval Augmented Generation (RAG) system.

## Assignment 2 (Day 2)

### Finetuning LLMs

Dive into finetuning LLMs such as the LLaMA 2 and LLaMA 3 models. You will explore two approaches: Full-finetuning and Low-Rank Adaptation (LoRA). Post-finetuning, evaluate the models using a separate data set to assess performance.

## Assignment 3 (Day 3)

### Assignment 3.1: Finetuning Multimodal LLMs

Learn to finetune Multimodal LLMs, such as MiniCPM-Llama3-V 2.5 or MiniCPM-V 2.0, using both Full-finetuning and LoRA approaches. After finetuning, evaluate these models on a distinct data set.

### Assignment 3.2: RAG Pipeline

Understand the Retrieval Augmented Generation (RAG) pipeline, where contexts are retrieved from vector databases to augment external knowledge. This enhances the LLM's ability to answer queries more effectively.

## Assignment 4 (Day 4)

### Model Serving

Explore various inference engines such as Fastchat, Ollama, VLLM, Tensor-RT, and ONNX. You will learn how to serve LLMs through these engines, enabling efficient model inference.

