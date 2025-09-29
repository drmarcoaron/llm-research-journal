This Jupyter Notebook, titled `activation-function.ipynb`, details an **experiment comparing the performance of different activation functions and the effect of attention bias** in a minimal Large Language Model (LLM).

Here is a summary of the notebook's purpose, experimental setup, and intended outputs:

### 1. Purpose of the Experiment

The primary goal of the notebook is to evaluate how three popular activation functions (ReLU, GELU, and SiLU) combined with the presence or absence of a bias term in the attention mechanism's linear layers affect a minimal GPT-style model's performance on a language modeling task.

### 2. Experimental Setup

The notebook defines a modular, minimal LLM architecture and runs a total of **six** distinct training experiments:

| Activation Function | Attention Bias | Experiment Name |
| :--- | :--- | :--- |
| **ReLU** | True | `relu_bias_True` |
| **ReLU** | False | `relu_bias_False` |
| **GELU** | True | `gelu_bias_True` |
| **GELU** | False | `gelu_bias_False` |
| **SiLU** | True | `silu_bias_True` |
| **SiLU** | False | `silu_bias_False` |

#### Model Architecture (`MinimalLLM`):
* **Type:** Decoder-only Transformer (GPT-style)
* **Layers:** 6 Transformer Blocks
* **Model Dimension (`d_model`):** 384
* **Attention Heads (`n_heads`):** 8
* **FFN Inner Dimension (`d_ff`):** 1536
* **Positional Encoding:** Uses Rotary Positional Embeddings (RoPE) in the `MultiHeadAttention` block.

#### Training and Data:
* **Dataset:** A subset of the "HuggingFaceTB/smollm-corpus" (specifically "cosmopedia-v2")
* **Max Tokens:** 500,000
* **Training Steps:** The accompanying report markdown in the code states the models are trained for 1000 steps.
* **Metrics Tracked:** Training Loss, Validation Loss, Validation Accuracy, and Validation Perplexity.

### 3. Key Functions and Logic

* **`FeedForward`:** This module is responsible for switching between the three activation functions: `F.relu`, `F.gelu`, or `F.silu`.
* **`MultiHeadAttention`:** The `use_attention_bias` flag determines if bias terms are used in the QKV and output linear layers (`self.qkv` and `self.w_o`).
* **`train_model`:** Manages the training loop, including mixed-precision training (`autocast`), gradient accumulation, gradient clipping, and a cosine-decay learning rate scheduler with warmup.
* **`evaluate_model`:** Calculates validation loss, accuracy, and perplexity.
* **`plot_results`:** Generates and saves four comparison plots in the `experiment_images` directory for each of the tracked metrics (Loss, Accuracy, Perplexity).
* **`generate_report`:** Produces a final `report.md` file. It determines the **best performing combination** by finding the experiment with the **lowest final validation loss** and includes this conclusion in the report.
