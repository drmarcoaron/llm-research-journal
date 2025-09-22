
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

**Model & Task:**
*   **Base Model:** `microsoft/DialogRPT-updown` (~80M parameters). Chosen for its suitability to rapid iteration.
*   **Task:** Sequence classification/ranking fine-tuning.
*   **Dataset:** A fixed, small subset of a relevant dataset (e.g., `AppReviews`).

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

## 5. Results and Analysis

*(This section will be populated with tables and graphs from your experiments)*

**Table 1: Summary of Key Metrics (Average of 3 Runs)**
| Optimizer | Peak VRAM (GB) | Throughput (samples/sec) | Time to Loss=X (min) | Final Loss | Stable? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| AdamW (baseline) |  |  |  |  | 3/3 |
| Lion |  |  |  |  | 3/3 |
| Sophia |  |  |  |  | 2/3 |
| Adam8bit |  |  |  |  | 3/3 |

**Figure 1: Validation Loss vs. Wall-Clock Time**
*(A line chart showing the learning curves, demonstrating which optimizer converges fastest in real-time)*

**Figure 2: Validation Loss vs. Training Step**
*(A line chart showing the sample efficiency, demonstrating which optimizer requires fewer update steps)*

**Figure 3: Memory Usage Over Time**
*(A plot showing the VRAM usage throughout the training process)*

## 6. Discussion and Practical Recommendations

Based on the results, we will provide a clear decision guide:

*   **Choose `Adam8bit` if:** Your primary constraint is **VRAM**. Use it to fit a larger model or a larger batch size.
*   **Choose `Lion` if:** You want a **memory-efficient** and **fast** optimizer with stability on par with AdamW.
*   **Choose `Sophia` if:** Your primary constraint is **time** (number of steps) and you are willing to tune it carefully for stability.
*   **Stick with `AdamW` if:** You prioritize **reliability and reproducibility** over marginal gains, or if your task is highly sensitive to hyperparameters.

## 7. Conclusion

This study demonstrates that for LoRA fine-tuning on constrained hardware, the choice of optimizer is not merely a theoretical concern but a practical engineering decision with significant implications for memory, speed, and stability. There is no single "best" optimizer; the optimal choice is a trade-off dictated by the specific constraints of the project.

## 8. Repository Structure
```
.
├── configs/                 # YAML files for each optimizer's training run
├── scripts/
│   ├── train.py            # Main training script
│   └── profile_memory.py   # Utility for measuring VRAM
├── notebooks/
│   ├── 01_optimizer_sweep.ipynb
│   ├── 02_analysis_plots.ipynb
│   └── 03_throughput_benchmark.ipynb
├── results/                # CSV files, plots, and final summaries
└── README.md
```

---

### Why This is Impressive to OpenAI Researchers:

1.  **Scientific Rigor:** You're not just running code; you're designing a controlled experiment. Isolating variables (LR tuning per optimizer) is key to valid results.
2.  **Practical Relevance:** The question is immediately useful for anyone training models, especially at a scale where efficiency matters. OpenAI researchers deal with massive compute budgets but also massive models, so efficiency is paramount.
3.  **Focus on Measurement:** You're proposing to measure the right things—especially **wall-clock time to convergence**, which is the ultimate metric for a practitioner, not just steps-to-convergence.
4.  **Understanding of Hardware Constraints:** Acknowledging and designing for the T4 constraint shows pragmatism and real-world problem-solving skills.
5.  **Clear Communication:** The structure is that of a mini-research paper. It tells a story: "Here's a question, here's how I'll answer it, here are my results, and here's what it means for you."

This is a perfect project. Execute this well, and it will be a stellar addition to your portfolio.
