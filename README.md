LLM Research Journal

This is my open notebook — a public lab journal where I run small but sharp experiments to understand how large language models really work under the hood. Each day I explore one fundamental building block of modern AI systems, document the results, and share everything openly.

The spirit: “Build, measure, learn, repeat.”

Core Experiments.

Activation Functions: Compare ReLU, GELU, SiLU, SwiGLU on a toy LM.

Mixed Precision (AMP): Benchmark FP32 vs AMP on Colab T4 across model sizes.

Mixture of Experts: Study tradeoff between few-big vs many-small experts.

Expert Capacity: Explore capacity factor (1.0 → 2.0) and overflow.

Wrap-Up: Combine best ideas into one “ultimate” model + write results summary.


Beyond the sprint, I will also investigate:

Fine-tuning Tradeoffs

Comparing LoRA vs QLoRA in low-resource settings.

Studying quantization during fine-tuning (impact on stability, convergence, and downstream accuracy).

Evaluating post-training quantization (PTQ) vs quantization-aware training (QAT).

Optimizer tradeoffs

Benchmark Lion, Sophia, AdamW, Adam8bit → effects on VRAM, throughput, stability.

Gradient Accumulation vs Batch Size

Finding the sweet spot for stable training on limited VRAM.

Efficient Attention

Implementing FlashAttention, sliding window, and simpler memory-efficient variants.

Distillation & Model Compression

Training small models to mimic larger ones for edge deployment.

 Deliverables

For each experiment I will publish:

Colab Notebook (fully reproducible)

Graph/Plot (loss curves, speedups, tradeoffs)

One-line Conclusion in README

Code & Results in /experiments and /results

