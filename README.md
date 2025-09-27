## What I focus on ?

* **Mixed precision & AMP** — measure speed/quality tradeoffs on T4/Colab; show stable recipes. 
* **Efficient attention** (FlashAttention, sliding-window) — implement, benchmark memory and throughput. 
* **Finetuning at low cost** (LoRA vs QLoRA + quant-aware finetune) — recipe + convergence graphs. 
* **Quantization & PTQ/QAT** — int8/AWQ/GGUF pipeline and accuracy vs size charts. 
* **Optimizers & memory tricks** (Lion, Adam8bit, gradient accumulation) — VRAM, speed, stability tradeoffs. 
* **Inference stack & KV cache** — port small model to Deepspeed / vLLM / ExLlama, measure latency and cost. 
* **Distillation & compression** — distill → export small model + deployment artifacts.
* **Mixture-of-Experts (toy)** — router, utilization plots, FLOP tradeoffs. 


