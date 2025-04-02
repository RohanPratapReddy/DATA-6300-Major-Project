# Backward Attention Fine-Tuning for GPT-2

## 📚 Project Title
**Enhancing Math Reasoning in GPT‑2 with a Novel Final Attention Head: A Fine‑Tuning Approach on OpenMathInstruct‑2**

## 👨‍💻 Authors
- Rohan Pratap Reddy Ravula — [ravular@wit.edu](mailto:ravular@wit.edu)  
- Annanahmed Furkanahmed Shaikh — [shaikha4@wit.edu](mailto:shaikha4@wit.edu)  
School of Computing and Data Science,  
Wentworth Institute of Technology, Boston, MA, USA

## 📄 Abstract
This project introduces a novel fine-tuning approach for transformer-based language models by latching a backward attention layer atop the frozen GPT-2 architecture. Instead of using a traditional linear layer to produce logits, we reweight token probabilities using an attention mechanism guided by the model's own output logits. This method improves math-based reasoning, reduces dependency on hyperparameter tuning (e.g., temperature, top-k), and ensures computational efficiency by updating only the added layer's parameters. The model is fine-tuned on NVIDIA's OpenMathInstruct-2 dataset using supervised learning.

## 🧠 Core Ideas
- Replace GPT-2’s final linear decoder with a **Backward Attention Head**.
- Introduce a **context-aware token reweighting mechanism** via a Key-Query-Value system.
- Use GPT-2's **logits to scale value vectors**, normalize, and project into a query.
- Compute attention scores against key vectors and apply Softmax to obtain token probabilities.
- **Only train the new attention head**; keep GPT-2 parameters frozen.

## 🔢 Dataset
- **OpenMathInstruct‑2** by NVIDIA, a high-quality dataset for mathematical reasoning tasks.
- Dataset Structure:
  - `problem` → Input prompt
  - `generated_solution + expected_answer` → Target sequence

## 🏗️ Model Architecture
1. Input embeddings are normalized and optionally projected into a latent space.
2. Key (K) and Value (V) vectors are computed from embeddings.
3. Logits are used to compute a weighted average of value vectors.
4. This average is normalized and projected into a query vector (Q).
5. Q is dotted with K to compute attention scores.
6. Softmax is applied to get output token probabilities.

## 🧪 Training
- Loss Function: Cross Entropy Loss
- Training Approach: Supervised learning with teacher forcing
- Only the backward attention module is updated.
- Code supports chunked attention and multi-head options for large vocabularies.

## 🧩 Modularity
- Fully modular design with interchangeable components:
  - RMSNorm, latent projection, attention scoring, softmax head
- Can be added/removed independently of the language model.
- Supports dynamic task-specific attachments for modular LLM assistants.

## 💻 Technologies Used
- Python 3.10+
- PyTorch
- Transformers (Hugging Face)
- CUDA (for GPU acceleration)

## 🔍 Evaluation
- Metrics:
  - Perplexity
  - Cross-Entropy Loss
  - Regression Metrics: MSE, RMSE
  - Classification Metrics: Accuracy, Precision, Recall, F1 Score
- Baselines:
  - GPT-2 fine-tuned with traditional methods
  - Llama3.1‑405B‑Instruct used in original dataset generation

## 🔬 Future Work
- Expand to multi-layer backward attention
- Integration into modular LLM pipelines
- Exploration of non-linear projection functions
- Application to multi-modal domains and few-shot tasks

## 🙏 Acknowledgements
- **Prof. Salem Othman** for consistent motivation, guidance, and encouragement throughout the project.
- **NVIDIA** for releasing the OpenMathInstruct‑2 dataset.
- **OpenAI** for the pretrained GPT‑2 model.
- **Hugging Face** for maintaining an excellent open-source platform.
- **Vaswani et al.** for the foundational paper _"Attention Is All You Need"_, which inspired the core innovation in this work.

## 📜 Citation
If you use or reference this work, please cite:
```bibtex
@misc{ravula2025backwardattention,
  author = {Rohan Pratap Reddy Ravula and Annanahmed Furkanahmed Shaikh},
  title = {Enhancing Math Reasoning in GPT-2 with a Novel Final Attention Head: A Fine-Tuning Approach on OpenMathInstruct-2},
  year = {2025},
  note = {Wentworth Institute of Technology, School of Computing and Data Science}
}
