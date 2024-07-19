# Assignment 2

## Question 1.1

The loss should decrease gradually but with some fluctuations.

## Question 1.2

- GPU memory shall increase as the batch size increases, but the time taken to train should decrease.
- Optimal batch size for LoRA should be 2.
  - When batch size is set to 1, the GPU memory usage is tested as below (default setting):
  ![image](https://github.com/user-attachments/assets/ff2f9633-5f1a-4944-a89a-04e0fd45b75b)
  - When batch size is set to 2, the GPU memory usage is tested as below:
  ![image](https://github.com/user-attachments/assets/caeb9fe3-7800-4c86-8fd5-60332ceeb6a5)


## Question 2

Inside the `finetune_lora.sh`, try changing, e.g., gradient_checkpointing and load_in_8bit, load_in_4bit.

Hyperparameters in the script:
- If deleted gradient_checkpointing, GPU would run out of memory.
![image](https://github.com/user-attachments/assets/8af344a4-c35c-4df6-8c36-25d47c447d19)
- If deleted load_in_8bit, GPU memory usage would increase (compared with default).
![image](https://github.com/user-attachments/assets/5b779db5-0220-4414-aa00-4d7858e7eaee)    

Additional hyperparameters:
- Change load_in_8bit to load_in_4bit, memory usage decreased greatly.
![image](https://github.com/user-attachments/assets/73ac3ac8-7172-4a49-b426-f8f8b57dce93)
    
## Question 3

Optimal batch size for FFT should be 6, FFT runs faster as more GPU used.

<br>

>(optional) If you tried to change the number of GPU used for FFT, you may explore something unexpected:   
When the number of GPUs used by LoRa and the number of GPUs used by FFT are both 1, and the batch_size is set to 1 for both, the GPU memory usage is shown below (GPU 0 for LoRA, GPU 1 for FFT).
![image](https://github.com/user-attachments/assets/6279cfbc-3020-47f5-8eb4-cc0c382affd6)
 > - The FFT uses less device memory than LoRA. This may seem counter-intuitive, but it is because the FFT uses a lot of host memory to perform the calculations.
  >- LoRA host memory usage:  
![image](https://github.com/user-attachments/assets/b58b6880-fdab-461e-9524-6c58a12d8845)
  >- FFT host memory usage:  
![image](https://github.com/user-attachments/assets/cc0eb493-bfe8-4874-ba92-5f42d786a7c0)

<br>

When FFT is trained with GPU number of 4, FFT uses less memory on a single device, but the total device memory usage exceeds that of LoRA.   
![image](https://github.com/user-attachments/assets/2f28eb2d-44db-4825-b687-412171a4cc30)
![image](https://github.com/user-attachments/assets/f330fe07-1927-4683-87c8-17e05427f1a0)

Full-parameter finetuning consumes a lot of memory because it updates all model parameters, requiring more GPU memory for gradients and updates. Additionally, a large batch size processes more data at once, which also increases memory usage.

## Question 4

By testing through the example ids, we can find that Llama3 typically has better performance than Llama2.

## Question 5

Through testing various results, we find fine-tuned models performing better than original models. Models fined-tuned by LoRA and FFT typically have similar performance.
