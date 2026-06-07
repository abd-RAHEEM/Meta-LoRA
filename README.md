<img width="1024" height="400" alt="Meta-LoRA Banner" src="https://github.com/user-attachments/assets/bd0b6db3-6d93-443a-aa7a-a9def7bd5527" />

<div align="center">

#  Meta-LoRA: Scaffold-Episodic Meta-Learning with Context-Conditioned LoRA for Few-Shot Molecular Generation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VH2OiE66KpyaCXJHiptN3jVkFy395kOx?usp=sharing)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20HF%20Spaces-Live%20Demo-yellow)](https://huggingface.co/spaces/abdulRaHeeM452/Molecule-Generator)
[![Model on HF](https://img.shields.io/badge/🤗%20Model-abdulRaHeeM452%2FMolecule--generator-blue)](https://huggingface.co/abdulRaHeeM452/Molecule-generator)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)

**Generate novel, valid, drug-like molecules from just 3–10 SMILES examples — zero gradient steps at inference.**

</div>

---

##  Overview

**Meta-LoRA** is a scaffold-aware few-shot molecular generation framework that combines:

- **Episodic meta-learning** over Bemis-Murcko scaffold families from ZINC250k
- **Context-conditioned LoRA** — rank-16 Low-Rank Adaptation injected into frozen transformer attention layers, dynamically conditioned on a support-set constraint vector
- **Dual-signal context encoding** — fusing GRU-based SMILES token sequences with Morgan fingerprint projections for richer structural representation

Given a small _support set_ of SMILES strings from the same scaffold family, the model instantly adapts its generation distribution at inference — with **no gradient updates** — to produce novel molecules sharing structural and pharmacophoric similarity.

---

##  Key Features

| Feature                     | Details                                                            |
| --------------------------- | ------------------------------------------------------------------ |
|  **Few-Shot**             | Works with as few as 3 SMILES strings                              |
|  **Zero-Shot Adaptation** | No fine-tuning at inference time                                   |
|  **Scaffold-Aware**       | Trained episodically on Bemis-Murcko scaffold families             |
|  **Frozen Base Model**    | 3.2M base params completely frozen; only ~500k LoRA params trained |
|  **Drug-Like Outputs**    | High QED, valid Lipinski properties                                |
|  **Rich Metrics**         | Validity, Uniqueness, Novelty, Tanimoto similarity, QED, LogP, MW  |

---

##  Architecture

```
Support Set (K SMILES)
        │
        ▼
┌─────────────────────────────────┐
│     EnhancedContextEncoder      │
│  ┌──────────┐  ┌─────────────┐  │
│  │ GRU over │  │  Morgan FP  │  │
│  │  SMILES  │  │  Projection │  │
│  │  tokens  │  │  (2048-bit) │  │
│  └────┬─────┘  └──────┬──────┘  │
│       └──────┬─────────┘         │
│          Fusion Linear           │
└──────────────┬──────────────────┘
               │  constraint vector z  (d=256)
               ▼
┌─────────────────────────────────┐
│   ContextConditionedLoRA        │
│   rank-16, applied to W_q & W_v │
│   in all 4 transformer layers   │
│   ΔW = B(z) @ A  (context-aware)│
└──────────────┬──────────────────┘
               │  LoRA deltas per layer
               ▼
┌─────────────────────────────────┐
│   BaseGrammarTransformer        │ ← FROZEN (3.2M params)
│   d=256, 4 layers, 8 heads      │
│   pre-trained on ZINC250k       │
│   Weight-tied embeddings        │
└──────────────┬──────────────────┘
               ▼
         Novel SMILES
```

### Component Breakdown

| Component                  | Description                                                                                        | Parameters        |
| -------------------------- | -------------------------------------------------------------------------------------------------- | ----------------- |
| **BaseGrammarTransformer** | 4-layer Pre-LN Transformer with weight-tied embeddings, pre-trained on 250k ZINC molecules         | ~3.2M (frozen)    |
| **EnhancedContextEncoder** | 2-layer GRU over SMILES tokens + Morgan FP projection, fused via linear layer                      | ~500k (trainable) |
| **ContextConditionedLoRA** | Rank-16 LoRA on Q and V projections of all 4 layers; B matrices generated from constraint vector z | ~500k (trainable) |
| **SMILESTokenizer**        | Regex-based tokenizer (Schwaller et al., 2019) handling multi-char tokens (Br, Cl, etc.)           | —                 |

---

##  Benchmark Results

Results from a **10-trial robustness experiment** on ZINC250k (5-shot, 100 molecules generated per trial, seed=42):

| Metric           | Mean ± Std     |
| ---------------- | -------------- |
| **Validity**     | 96.8 ± 2.1%    |
| **Uniqueness**   | 99.1 ± 0.8%    |
| **Novelty**      | 98.3 ± 1.4%    |
| **Avg Tanimoto** | 0.4231 ± 0.038 |

> Each trial uses a **different, independently sampled** Bemis-Murcko scaffold family — proving robustness across diverse chemical space.

### Ablation Study

| Condition                                      | Validity  | Tanimoto  |
| ---------------------------------------------- | --------- | --------- |
| A — Base model, unconditional                  | ~96%      | ~0.15     |
| B — Naive fine-tuning (10 steps, all weights)  | collapses | ~0.20     |
| C — Our model, SMILES tokens only (no FP)      | ~95%      | ~0.38     |
| **D — Full model (tokens + Morgan FP + LoRA)** | **~97%**  | **~0.42** |

The ablation confirms that:

- Fine-tuning all weights leads to catastrophic forgetting of chemical grammar
- The Morgan fingerprint branch provides a meaningful structural signal (+0.04 Tanimoto)
- Context-conditioned LoRA successfully steers generation without any gradient updates

---

##  Live Demo

Try the interactive Gradio app on Hugging Face Spaces:

👉 **[Meta-LoRA Molecular Generator — HF Spaces](https://huggingface.co/spaces/abdulRaHeeM452/Molecule-Generator)**

Paste 3–10 SMILES from the same scaffold family, adjust temperature and number of molecules, and click **Generate**. The app visualizes the generated molecules and reports QED, LogP, MW, H-bond donors/acceptors.

---

##  Training Pipeline

The full training pipeline is in the Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VH2OiE66KpyaCXJHiptN3jVkFy395kOx?usp=sharing)

### Phase 1 — Base Grammar Pre-training

- Dataset: **ZINC250k** (250k drug-like molecules from the ZINC database)
- Model: `BaseGrammarTransformer` (d=256, 4 layers, 8 heads, Pre-LN)
- Training: 10 epochs, AdamW (lr=3e-4), cosine LR schedule with 500-step warmup, label smoothing=0.1
- Loss: Cross-entropy on next-token prediction with causal masking

### Phase 2 — Scaffold-Episodic Meta-Training

- Scaffold families constructed via **Bemis-Murcko** decomposition (min 15 members per family)
- Each episode: randomly sample a scaffold family → draw K=5 support molecules + 32 query molecules
- Support set encodes dual signal: SMILES token GRU + 2048-bit Morgan fingerprint (radius=2)
- Only the **ContextEncoder** and **LoRA** adapters are trained (~500k params); base model stays frozen
- Training: 10 epochs × 1000 episodes, AdamW (lr=3e-4), cosine annealing

---

##  Using the Model Programmatically

The deployed model is hosted on Hugging Face Hub at [`abdulRaHeeM452/Molecule-generator`](https://huggingface.co/abdulRaHeeM452/Molecule-generator).

```python
from inference import load_models, run_generation

# Load models (downloads from HF Hub on first run)
load_models()

# Define your support set — 3 to 10 SMILES from the same scaffold family
support_smiles = [
    "CC(=O)Oc1ccccc1C(=O)O",   # aspirin
    "CC(=O)Oc1ccc(C)cc1C(=O)O",
    "CC(=O)Oc1cccc(C(=O)O)c1",
    "CC(=O)Oc1ccc(F)cc1C(=O)O",
    "CC(=O)Oc1ccc(Cl)cc1C(=O)O",
]

# Generate molecules
results = run_generation(support_smiles, n=50, temperature=0.8)

print(f"Validity  : {results['validity']:.1f}%")
print(f"Uniqueness: {results['uniqueness']:.1f}%")
print(f"Novelty   : {results['novelty']:.1f}%")
print(f"Tanimoto  : {results['avg_tanimoto']:.4f}")

# Access generated SMILES
for smi in results['novel_smiles'][:5]:
    print(smi)
```

### Generation Parameters

| Parameter        | Default | Description                                                             |
| ---------------- | ------- | ----------------------------------------------------------------------- |
| `n`              | 50      | Number of molecules to generate (10–200)                                |
| `temperature`    | 0.8     | Sampling temperature — lower = more conservative, higher = more diverse |
| Support set size | 5       | 3–10 SMILES strings from the same scaffold family                       |

---

##  Repository Structure

```
Meta-LoRA/
├── Scaffold_Episodic_Meta_Learning_...ipynb   # Full training & evaluation notebook (Colab)
├── scaffold_episodic_meta_learning_....py     # Python export of the notebook
├── inference.py                               # Self-contained inference module (load & run locally)
├── requirements.txt                           # Python dependencies
└── README.md
```

> **Note on `inference.py`**: This is a self-contained inference module — it downloads model weights from [HF Hub](https://huggingface.co/abdulRaHeeM452/Molecule-generator) on first run and exposes a clean `run_generation()` API. The Gradio UI (`app.py`) lives exclusively on HF Spaces and is not tracked here to keep the repository focused on the science. Model weights (`grammar_engine_v3.pt`, `meta_engine_v3.pt`, `smiles_tokenizer_v3.json`) are versioned separately on HF Hub.

---

##  Local Setup

```bash
# Clone the repo
git clone https://github.com/abd-RAHEEM/Meta-LoRA.git
cd Meta-LoRA

# Install dependencies
pip install -r requirements.txt

# Run the Gradio app locally
python app.py
```

> **GPU recommended** — the model runs on CPU too, but generation will be slower.

---

##  Dependencies

| Package          | Version    |
| ---------------- | ---------- |
| PyTorch          | ≥ 2.1.0    |
| RDKit            | ≥ 2023.3.1 |
| Gradio           | 5.9.1      |
| Hugging Face Hub | ≥ 0.20.0   |
| NumPy            | ≥ 1.24.0   |
| Pandas           | ≥ 2.0.0    |
| Pillow           | ≥ 10.0.0   |

---

## 🔗 Links

| Resource                     | Link                                                                                                        |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------- |
|  Training Notebook (Colab) | [Open in Colab](https://colab.research.google.com/drive/1VH2OiE66KpyaCXJHiptN3jVkFy395kOx?usp=sharing)      |
|  Live Demo (HF Spaces)     | [Meta-LoRA Molecular Generator](https://huggingface.co/spaces/abdulRaHeeM452/Molecule-Generator) |
|  Model Weights (HF Hub)    | [abdulRaHeeM452/Molecule-generator](https://huggingface.co/abdulRaHeeM452/Molecule-generator)               |
|  GitHub Repository         | [abd-RAHEEM/Meta-LoRA](https://github.com/abd-RAHEEM/Meta-LoRA)                                             |

---

##  References

- Hu, E. J. et al. (2022). **LoRA: Low-Rank Adaptation of Large Language Models.** _ICLR 2022._
- Bemis, G. W. & Murcko, M. A. (1996). **The Properties of Known Drugs. 1. Molecular Frameworks.** _J. Med. Chem._
- Schwaller, P. et al. (2019). **Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction.** _ACS Cent. Sci._
- Irwin, J. J. & Shoichet, B. K. (2005). **ZINC — A Free Database of Commercially Available Compounds for Virtual Screening.** _J. Chem. Inf. Model._
- Press, O. & Wolf, L. (2017). **Using the Output Embedding to Improve Language Models.** _EACL 2017._

---

##  License

This project is released under the [MIT License](LICENSE).

---

<div align="center">
Made with ❤️ for computational drug discovery
</div>
