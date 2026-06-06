---
title: Meta-LoRA Molecular Generator
emoji: ⚗️
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
python_version: "3.11"
pinned: false
license: mit
short_description: Few-shot molecular generation via context-conditioned LoRA
---

# Meta-LoRA Molecular Generator

Scaffold-Episodic Meta-Learning with Context-Conditioned LoRA for Few-Shot Molecular Generation.

Given a small support set of SMILES strings from the same scaffold family, the model generates
novel, valid, drug-like molecules — **zero gradient steps at inference**.

## Architecture
- **BaseGrammarTransformer** — 3.2M frozen params, pre-trained on ZINC250k
- **EnhancedContextEncoder** — GRU + Morgan FP fusion → constraint vector z
- **ContextConditionedLoRA** — rank-16 LoRA on Q and V projections, conditioned on z
- **Training** — Scaffold-episodic meta-training (Bemis-Murcko families)

## Metrics (10-trial mean ± std, ZINC250k, 5-shot)
| Metric | Value |
|---|---|
| Validity | 96.8 ± 2.1% |
| Uniqueness | 99.1 ± 0.8% |
| Novelty | 98.3 ± 1.4% |
| Avg Tanimoto | 0.4231 ± 0.038 |

## Usage
Paste 3–10 SMILES from the same scaffold family into the support set box and click Generate.
