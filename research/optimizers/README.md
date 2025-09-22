
---

# Optimizer Tradeoffs in Efficient Fine-Tuning: A Empirical Study of Lion, Sophia, AdamW, and Adam8bit

## Abstract

This study presents a controlled empirical comparison of modern optimization algorithms—**Lion**, **Sophia**, **AdamW**, and **8-bit Adam (Adam8bit)**—for parameter-efficient fine-tuning (PEFT) using Low-Rank Adaptation (LoRA). Conducted under the stringent memory and compute constraints of a single NVIDIA T4 GPU (16GB VRAM), we quantify the trade-offs between memory footprint, throughput, convergence speed, and training stability. Our goal is to provide a practical, data-driven "rulebook" for researchers and practitioners selecting an optimizer for fine-tuning large language models on limited hardware.

## 1. Introduction: The Optimizer's Role in constrained Fine-Tuning

While LoRA dramatically reduces the number of trainable parameters, the choice of optimizer remains critical. The optimizer governs the update rule and, crucially, determines the **memory overhead** of its state tensors (e.g., momentum, variance estimates). This overhead can be the limiting factor when fine-tuning large models on consumer or free-tier hardware.

This project moves beyond anecdotal evidence by providing a rigorous, apples-to-apples comparison of four promising optimizers for this use case, answering a simple question: *For a fixed compute budget and hardware constraint, which optimizer delivers the best performance?*

## 2. Optimizers Under Investigation

| Optimizer | Key Idea | Memory Footprint (per parameter) | Expected Advantage |
| :--- | :--- | :--- | :--- |
| **AdamW** | Adaptive learning rates with decoupled weight decay. | ~12 bytes (m, v, p) | The reliable baseline. Robust and well-understood. |
| **Lion** | Sign-based updates with momentum. Simpler than Adam. | ~8 bytes (m, p) | **Lower memory** and faster step computation. |
| **Sophia** | Second-order-inspired (Hessian-aware) with clipping. | ~12 bytes (m, v, p) | **Faster convergence** (fewer steps to target loss). |
| **Adam8bit** | AdamW with 8-bit state quantization. | ~6 bytes (m, v, p) | **Lowest memory**, enabling larger models/batches. |

## 3. Experimental Design

To ensure fair and reproducible results, we fix all variables except the optimizer.


**LoRA Configuration:**
*   **Rank (r):** 8
*   **Alpha:** 16
*   **Target Modules:** `q_proj, v_proj` (standard for LLMs)

**Training Hyperparameters:**
*   **Batch Size:** Maximized for each optimizer within the 16GB VRAM limit.
*   **Learning Rate:** Individually tuned via a minimal sweep for each optimizer to find its stable maximum.
*   **Max Steps:** 1000 (or until clear convergence observed).
*   **Seeds:** Each experiment run 3 times with different seeds to measure variance.

## 4. Metrics: What We Measure and Why

1.  **Memory Efficiency:**
    *   `Peak VRAM Allocation`: The maximum GPU memory allocated during training (via `torch.cuda.max_memory_allocated()`).
    *   `Optimizer State Size`: The total size of all optimizer tensors, calculated directly from the optimizer's `state_dict`.

2.  **Computational Efficiency:**
    *   `Throughput`: Measured in **samples/second** and **tokens/second**.
    *   `Time-to-Convergence`: Wall-clock time to achieve a target validation loss (e.g., 1.5, 1.2, 1.0).

3.  **Performance & Stability:**
    *   `Final Validation Loss`: Primary performance metric after a fixed number of steps.
    *   `Training Dynamics`: Smoothed loss curves and gradient norm plots to assess stability.
    *   `Divergence Count`: The number of runs (out of 3) where training becomes unstable (NaN loss).



## 7. Conclusion

This study demonstrates that for LoRA fine-tuning on constrained hardware, the choice of optimizer is not merely a theoretical concern but a practical engineering decision with significant implications for memory, speed, and stability. There is no single "best" optimizer; the optimal choice is a trade-off dictated by the specific constraints of the project.

