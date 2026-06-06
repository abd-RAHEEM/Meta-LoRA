"""
app.py — Gradio interface for Meta-LoRA Molecular Generator
Deploy this on Hugging Face Spaces (SDK: Gradio)
"""

import gradio as gr
import pandas as pd
from inference import run_generation, load_models, mol_to_pil

# Pre-load models when Space boots (not on first user request)
load_models()

# ── Example scaffold families (paste-ready) ────────────────────────────────────
EXAMPLES = {
    "Aspirin-like (salicylates)": "\n".join([
        "CC(=O)Oc1ccccc1C(=O)O",
        "CC(=O)Oc1ccc(C)cc1C(=O)O",
        "CC(=O)Oc1cccc(C(=O)O)c1",
        "CC(=O)Oc1ccc(F)cc1C(=O)O",
        "CC(=O)Oc1ccc(Cl)cc1C(=O)O",
    ]),
    "Caffeine-like (xanthines)": "\n".join([
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
        "O=c1[nH]cnc2c1ncn2C",
        "Cn1cnc2c1c(=O)n(CC)c(=O)n2C",
        "Cn1cnc2c1c(=O)nc(=O)n2CC",
    ]),
    "Ibuprofen-like (arylpropionic)": "\n".join([
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "CC(C)Cc1ccc(C(C)C(=O)OC)cc1",
        "CC(C(=O)O)c1ccc(Cl)cc1",
        "CC(C(=O)O)c1cccc(F)c1",
        "CC(C(=O)O)c1ccc(CC)cc1",
    ]),
}


def load_example(choice):
    return EXAMPLES.get(choice, "")


def validate_smiles_input(smiles_text: str):
    from rdkit import Chem
    lines = [s.strip() for s in smiles_text.strip().split('\n') if s.strip()]
    if len(lines) < 3:
        return None, "❌ Please enter at least 3 SMILES strings (one per line)."
    if len(lines) > 10:
        return None, "❌ Maximum 10 SMILES in the support set."
    valid = []
    for smi in lines:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None, f"❌ Invalid SMILES: `{smi}`"
        valid.append(smi)
    return valid, None


def generate_molecules(smiles_text: str, n_generate: int, temperature: float, progress=gr.Progress()):
    valid_smiles, err = validate_smiles_input(smiles_text)
    if err:
        return None, err, None, None

    progress(0.1, desc="Encoding support set...")
    try:
        progress(0.3, desc="Generating molecules...")
        results = run_generation(valid_smiles, n=int(n_generate), temperature=float(temperature))
    except Exception as e:
        return None, f"❌ Generation failed: {str(e)}", None, None

    progress(0.85, desc="Computing metrics...")

    # Build metrics summary card
    m = results
    summary_md = f"""
### Generation Results

| Metric | Value |
|---|---|
| Generated | {m['n_generated']} |
| ✅ Valid | {m['n_valid']} ({m['validity']:.1f}%) |
| 🔁 Unique | {m['n_unique']} ({m['uniqueness']:.1f}%) |
| ✨ Novel | {m['n_novel']} ({m['novelty']:.1f}%) |
| 🧬 Avg Tanimoto | {m['avg_tanimoto']:.4f} |

*Novel molecules shown below (up to 20)*
"""

    # Build molecule table for the dataframe
    rows = []
    for item in results['images']:
        props = item['props'] or {}
        rows.append({
            "SMILES": item['smiles'],
            "QED": props.get("qed", "N/A"),
            "LogP": props.get("logp", "N/A"),
            "MW": props.get("mw", "N/A"),
            "HBD": props.get("hbd", "N/A"),
            "HBA": props.get("hba", "N/A"),
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["SMILES","QED","LogP","MW","HBD","HBA"])

    # Image gallery: list of PIL images
    gallery = [item['image'] for item in results['images']]

    progress(1.0, desc="Done!")
    return summary_md, "", gallery, df


# ── UI ─────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Meta-LoRA Molecular Generator",
    theme=gr.themes.Soft(primary_hue="purple", secondary_hue="teal"),
) as demo:

    gr.Markdown("""
# ⚗️ Meta-LoRA Molecular Generator
**Scaffold-Episodic Meta-Learning with Context-Conditioned LoRA for Few-Shot Molecular Generation**

Paste 3–10 SMILES from the same scaffold family as a *support set*.  
The model generates novel molecules that share structural similarity — with **zero gradient steps at inference**.
""")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔬 Support Set")
            example_dropdown = gr.Dropdown(
                choices=list(EXAMPLES.keys()),
                label="Load an example scaffold family",
                value=None,
            )
            smiles_input = gr.Textbox(
                label="SMILES strings (one per line, 3–10)",
                placeholder="CC(=O)Oc1ccccc1C(=O)O\nCC(=O)Oc1ccc(C)cc1...",
                lines=8,
            )

            with gr.Row():
                n_generate  = gr.Slider(10, 200, value=50, step=10,
                                        label="Molecules to generate")
                temperature = gr.Slider(0.5, 1.5, value=0.8, step=0.05,
                                        label="Temperature")

            generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
            error_box    = gr.Markdown(visible=True)

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results")
            metrics_md  = gr.Markdown()
            mol_gallery = gr.Gallery(
                    label="Generated molecules (novel, valid)",
                    columns=4,
                    height="auto",
                    object_fit="contain",
                    )

    gr.Markdown("### 🧪 Molecule Properties")
    props_table = gr.Dataframe(
        headers=["SMILES","QED","LogP","MW","HBD","HBA"],
        interactive=False,
    )

    gr.Markdown("""
---
**Model:** Scaffold-Episodic Meta-Learning · Context-Conditioned LoRA (rank 16) · ZINC250k  
**Paper metrics:** Validity ~96.8% · Uniqueness ~99% · Novelty ~98% · Tanimoto ~0.42
""")

    # Wire events
    example_dropdown.change(fn=load_example, inputs=example_dropdown, outputs=smiles_input)
    generate_btn.click(
        fn=generate_molecules,
        inputs=[smiles_input, n_generate, temperature],
        outputs=[metrics_md, error_box, mol_gallery, props_table],
    )

demo.launch()
