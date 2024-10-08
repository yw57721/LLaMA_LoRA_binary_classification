# Parameter efficient fine-tuning (PEFT) LLaMA-3.1-8b using LoRA for Binary classification
This project aims to identify novel customer needs from user generated content, specifically online product reviews. We leverage LoRA to fine-tune LLaMA-3.1-8b in a parameter efficient way to incorporate domain knowledge of the target product to the LLM (i.e., LLaMA-3.1-8b). Laptops' reviews and the corresponding product metadata, such as the specifications were crawled from amazon.com and used to LoRA-fine-tune LLaMA-3.1-8b. Experiments demonstrate that LoRA-fine-tuned LLaMA achieves better results than the vanilla LLaMA. This shows the effectiveness of LoRA in creating domain specific LLM from general purpose LLM. 

To run the code, you'll need to obtain an LLaMA access token. Follow these easy steps:

1. Head to the official LLaMA repository on Hugging Face: [https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct).
2. Follow the on-screen instructions to get your LLaMA access token.

Once you have your access token, simply insert it into the Python code at line 11, where you'll see the placeholder text "llama_token = '[insert-llama-access-token-here]'. This will enable you to run the code successfully.
