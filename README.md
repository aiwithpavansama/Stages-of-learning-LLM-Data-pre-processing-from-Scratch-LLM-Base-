# 📖 Build an LLM from Scratch: Project Progress

This project tracks our step-by-step implementation of a Large Language Model (LLM) from scratch using PyTorch and tokenization tools.

---

## 🛠️ Completed Steps

### 📦 Stage 1: Data Preparation
1. **Data Loading**: Extracted raw text from a PDF source (`The_Verdict.pdf`) containing **21,813 characters** using `pypdf`.
2. **Text Tokenization & Vocabulary Building**:
   * Implemented custom regex-based text splitters to preserve punctuation.
   * Developed a custom `SimpleTokenizer` supporting special tokens (`<|unk|>` and `<|endoftext|>`).
   * Migrated to **`tiktoken`** (OpenAI's Byte Pair Encoding tokenizer) using the GPT-2 encoding vocabulary (**50,257** unique tokens).
3. **Dataset & DataLoader Pipeline**:
   * Engineered a custom PyTorch `Dataset` using a sliding window strategy to create contiguous overlapping input-target training pairs.
   * Integrated PyTorch's `DataLoader` to manage batch processing, shuffling, and shape adjustments.
4. **Dense Representation (Embeddings)**:
   * Established a trainable `nn.Embedding` matrix projecting discrete token IDs into dense **256-dimensional** vector spaces.
   * Implemented **Trainable Absolute Positional Embeddings** to inject token order representations, summing positional vectors directly with token semantic vectors.

---

## 🚀 Upcoming Step

### 🧠 Stage 2: Attention Mechanism
We are now moving on to the core engine of the Transformer architecture:
* **Self-Attention Mechanism**: Enabling tokens to dynamically weigh their relationships with every other token in the context window.
* **Scaled Dot-Product Attention**: Formulating the Key, Query, and Value matrix interactions.
* **Causal/Masked Attention**: Implementing the causal mask to ensure next-token prediction restricts access to future tokens.
