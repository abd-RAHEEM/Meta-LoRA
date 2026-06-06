# inference.py — standalone inference module, no training code
# Drop this file into your HF Space or any deployment.

import re, json, math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw, Descriptors, QED
from rdkit import DataStructs
from huggingface_hub import hf_hub_download
from PIL import Image

RDLogger.DisableLog('rdApp.*')

# ── Constants (must match training) ───────────────────────────────────────────
MAX_LEN   = 60
FP_DIM    = 2048
FP_RADIUS = 2

SMILES_REGEX = re.compile(
    r"(\[[^\[\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)

# ── Tokenizer ──────────────────────────────────────────────────────────────────
class SMILESTokenizer:
    def __init__(self):
        self.special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        self.vocab          = {t: i for i, t in enumerate(self.special_tokens)}
        self.inverse_vocab  = {i: t for t, i in self.vocab.items()}
        self.is_fit         = False

    def _tokenize(self, smiles: str):
        return SMILES_REGEX.findall(smiles)

    def load(self, path: str):
        with open(path, 'r') as f:
            self.vocab = json.load(f)
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
        self.is_fit = True

    def encode(self, smiles: str, max_length: int = MAX_LEN) -> list:
        if not self.is_fit:
            raise ValueError("Tokenizer not fit.")
        enc = [self.vocab['<SOS>']]
        for tok in self._tokenize(smiles):
            enc.append(self.vocab.get(tok, self.vocab['<UNK>']))
        enc.append(self.vocab['<EOS>'])
        if len(enc) > max_length:
            enc = enc[:max_length - 1] + [self.vocab['<EOS>']]
        enc += [self.vocab['<PAD>']] * (max_length - len(enc))
        return enc

    def decode(self, token_ids) -> str:
        out = ""
        for tid in token_ids:
            tok = self.inverse_vocab.get(int(tid), '<UNK>')
            if tok == '<EOS>':
                break
            if tok not in self.special_tokens:
                out += tok
        return out


# ── Base Grammar Transformer ───────────────────────────────────────────────────
class BaseGrammarTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8,
                 num_layers=4, max_seq_length=MAX_LEN):
        super().__init__()
        self.d_model    = d_model
        self.num_layers = num_layers
        self.token_embedding    = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.fc_out.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.zeros_(self.fc_out.bias)

    def generate_causal_mask(self, sz, device):
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, x, src_key_padding_mask=None):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        emb  = self.token_embedding(x) * math.sqrt(self.d_model)
        emb  = emb + self.position_embedding(pos)
        mask = self.generate_causal_mask(T, x.device)
        out  = self.transformer(emb, mask=mask,
                                src_key_padding_mask=src_key_padding_mask,
                                is_causal=True)
        return self.fc_out(out)


# ── Enhanced Context Encoder ───────────────────────────────────────────────────
class EnhancedContextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, fp_dim=FP_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.gru       = nn.GRU(d_model, d_model, batch_first=True,
                                num_layers=2, dropout=0.1)
        self.fp_proj   = nn.Sequential(
            nn.Linear(fp_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.fusion = nn.Linear(d_model * 2, d_model)

    def forward(self, support_set, support_fps=None):
        B, K, L = support_set.shape
        x        = support_set.reshape(B * K, L)
        x        = self.embedding(x)
        _, h     = self.gru(x)
        h        = h[-1].reshape(B, K, -1)
        seq_z    = h.mean(dim=1)
        fp_z     = self.fp_proj(support_fps.mean(dim=1)) if support_fps is not None else seq_z
        return self.fusion(torch.cat([seq_z, fp_z], dim=-1))


# ── Context-Conditioned LoRA ───────────────────────────────────────────────────
class ContextConditionedLoRA(nn.Module):
    def __init__(self, d_model=256, num_layers=4, rank=16, nhead=8):
        super().__init__()
        self.d_model    = d_model
        self.num_layers = num_layers
        self.rank       = rank
        self.nhead      = nhead
        self.head_dim   = d_model // nhead
        self.A_q = nn.ParameterList([nn.Parameter(torch.empty(rank, d_model)) for _ in range(num_layers)])
        self.A_v = nn.ParameterList([nn.Parameter(torch.empty(rank, d_model)) for _ in range(num_layers)])
        self.B_q_gen = nn.ModuleList([nn.Linear(d_model, d_model * rank) for _ in range(num_layers)])
        self.B_v_gen = nn.ModuleList([nn.Linear(d_model, d_model * rank) for _ in range(num_layers)])
        self.scaling = 0.5
        for i in range(num_layers):
            nn.init.kaiming_uniform_(self.A_q[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.A_v[i], a=math.sqrt(5))
            nn.init.zeros_(self.B_q_gen[i].weight)
            nn.init.zeros_(self.B_v_gen[i].weight)

    def get_delta_weights(self, z, layer_idx):
        B    = z.size(0)
        B_q  = self.B_q_gen[layer_idx](z).view(B, self.d_model, self.rank)
        B_v  = self.B_v_gen[layer_idx](z).view(B, self.d_model, self.rank)
        A_q  = self.A_q[layer_idx].unsqueeze(0).expand(B, -1, -1)
        A_v  = self.A_v[layer_idx].unsqueeze(0).expand(B, -1, -1)
        dW_q = torch.bmm(B_q, A_q) * self.scaling
        dW_v = torch.bmm(B_v, A_v) * self.scaling
        return dW_q, dW_v


# ── Rapid Adaptation Engine ────────────────────────────────────────────────────
class RapidAdaptationEngine(nn.Module):
    def __init__(self, base_model, context_encoder, d_model=256, pad_idx=0):
        super().__init__()
        self.base_model      = base_model
        self.context_encoder = context_encoder
        self.pad_idx         = pad_idx          # stored so forward() doesn't need global tokenizer
        self.lora            = ContextConditionedLoRA(
            d_model    = d_model,
            num_layers = base_model.num_layers,
            rank       = 16,
            nhead      = base_model.transformer.layers[0].self_attn.num_heads
        )
        for p in self.base_model.parameters():
            p.requires_grad = False

    def _lora_attention_forward(self, layer, x, dW_q, dW_v, attn_mask, key_padding_mask):
        B, T, d = x.shape
        sa      = layer.self_attn
        d_model = sa.embed_dim
        nhead   = sa.num_heads
        x_norm  = layer.norm1(x)
        W_qkv   = sa.in_proj_weight
        b_qkv   = sa.in_proj_bias
        W_q_frz = W_qkv[:d_model]
        W_k_frz = W_qkv[d_model:2*d_model]
        W_v_frz = W_qkv[2*d_model:]
        Q = torch.einsum('btd,bde->bte', x_norm, (W_q_frz + dW_q).transpose(1, 2))
        K = x_norm @ W_k_frz.T
        V = torch.einsum('btd,bde->bte', x_norm, (W_v_frz + dW_v).transpose(1, 2))
        if b_qkv is not None:
            Q = Q + b_qkv[:d_model]
            K = K + b_qkv[d_model:2*d_model]
            V = V + b_qkv[2*d_model:]
        head_dim = d_model // nhead
        scale    = head_dim ** -0.5
        Q = Q.view(B, T, nhead, head_dim).transpose(1, 2)
        K = K.view(B, T, nhead, head_dim).transpose(1, 2)
        V = V.view(B, T, nhead, head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        drop_p       = layer.self_attn.dropout if layer.training else 0.0
        attn_weights = F.dropout(attn_weights, p=drop_p)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, d_model)
        attn_out = F.linear(attn_out, sa.out_proj.weight, sa.out_proj.bias)
        x = x + attn_out
        drop_ffn = layer.dropout.p if hasattr(layer.dropout, 'p') else 0.0
        x = x + layer.linear2(
            F.dropout(layer.activation(layer.linear1(layer.norm2(x))), p=drop_ffn)
        )
        return x

    def forward(self, target_x, support_set, support_fps=None):
        z           = self.context_encoder(support_set, support_fps)
        B, T        = target_x.shape
        pos         = torch.arange(T, device=target_x.device).unsqueeze(0)
        x           = (self.base_model.token_embedding(target_x) *
                       math.sqrt(self.base_model.d_model)
                       + self.base_model.position_embedding(pos))
        pad_mask    = (target_x == self.pad_idx)
        causal_mask = self.base_model.generate_causal_mask(T, target_x.device)
        for i, frozen_layer in enumerate(self.base_model.transformer.layers):
            dW_q, dW_v = self.lora.get_delta_weights(z, i)
            x = self._lora_attention_forward(frozen_layer, x, dW_q, dW_v,
                                             attn_mask=causal_mask,
                                             key_padding_mask=pad_mask)
        return self.base_model.fc_out(x)


# ── Utilities ──────────────────────────────────────────────────────────────────
def smiles_to_fp(smiles: str, radius: int = FP_RADIUS, nbits: int = FP_DIM) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def mol_to_pil(smiles: str, size=(300, 300)) -> Image.Image | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


def get_drug_props(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        sa_score = _compute_sa_score(mol)
    except Exception:
        sa_score = None
    return {
        "qed":   round(QED.qed(mol), 4),
        "logp":  round(Descriptors.MolLogP(mol), 4),
        "mw":    round(Descriptors.MolWt(mol), 2),
        "hbd":   Descriptors.NumHDonors(mol),
        "hba":   Descriptors.NumHAcceptors(mol),
        "sa":    round(sa_score, 4) if sa_score else "N/A",
    }


def _compute_sa_score(mol):
    """SA score via RDKit contrib (if available), else None."""
    try:
        from rdkit.Contrib.SA_Score import sascorer
        return sascorer.calculateScore(mol)
    except ImportError:
        return None


# ── Generation ─────────────────────────────────────────────────────────────────
def generate_raw(model, tokenizer, support_smiles,
                 num_generate=50, max_length=MAX_LEN, temperature=0.8):
    model.eval()
    dev = next(model.parameters()).device

    sup_fps    = torch.tensor(
        np.array([smiles_to_fp(s) for s in support_smiles], dtype=np.float32)
    ).unsqueeze(0).expand(num_generate, -1, -1).to(dev)

    sup_enc    = [tokenizer.encode(s, max_length=max_length) for s in support_smiles]
    sup_tensor = torch.tensor(sup_enc, dtype=torch.long).unsqueeze(0) \
                      .expand(num_generate, -1, -1).to(dev)

    sos_id = tokenizer.vocab['<SOS>']
    eos_id = tokenizer.vocab['<EOS>']
    pad_id = tokenizer.vocab['<PAD>']
    unk_id = tokenizer.vocab['<UNK>']

    seqs    = torch.full((num_generate, 1), sos_id, dtype=torch.long, device=dev)
    eos_hit = torch.zeros(num_generate, dtype=torch.bool, device=dev)

    with torch.no_grad():
        for _ in range(max_length - 1):
            logits = model(seqs, sup_tensor, sup_fps)
            nxt    = logits[:, -1, :] / temperature
            for bad_id in [pad_id, sos_id, unk_id]:
                nxt[:, bad_id] = float('-inf')
            probs = F.softmax(nxt, dim=-1)
            tok   = torch.multinomial(probs, num_samples=1)
            tok[eos_hit] = eos_id
            eos_hit |= tok.squeeze(1) == eos_id
            seqs = torch.cat([seqs, tok], dim=1)
            if eos_hit.all():
                break

    return [tokenizer.decode(s.cpu().numpy()) for s in seqs]


def evaluate_generation(generated_smiles, support_smiles, verbose=False):
    valid_mols, canonical_gen = [], []
    for smi in generated_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_mols.append(mol)
            canonical_gen.append(Chem.MolToSmiles(mol))

    validity   = len(valid_mols) / max(1, len(generated_smiles)) * 100
    unique_can = list(set(canonical_gen))
    uniqueness = len(unique_can) / max(1, len(valid_mols)) * 100

    can_support = set()
    for smi in support_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            can_support.add(Chem.MolToSmiles(mol))

    novel    = [s for s in unique_can if s not in can_support]
    novelty  = len(novel) / max(1, len(unique_can)) * 100

    sup_fps = []
    for smi in can_support:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            sup_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_DIM))

    max_sims = []
    for smi in novel:
        mol_g = Chem.MolFromSmiles(smi)
        fp_g  = AllChem.GetMorganFingerprintAsBitVect(mol_g, FP_RADIUS, nBits=FP_DIM)
        sims  = [DataStructs.TanimotoSimilarity(fp_g, fp_s) for fp_s in sup_fps]
        max_sims.append(max(sims) if sims else 0.0)

    avg_tan = float(np.mean(max_sims)) if max_sims else 0.0

    return dict(
        validity=validity, uniqueness=uniqueness, novelty=novelty,
        avg_tanimoto=avg_tan, novel_smiles=novel,
        valid_mols=valid_mols, max_sims=max_sims,
        n_generated=len(generated_smiles), n_valid=len(valid_mols),
        n_unique=len(unique_can), n_novel=len(novel),
    )


# ── Global model state (loaded once at startup) ────────────────────────────────
_tokenizer  = None
_meta_model = None
_device     = None

HF_REPO_ID = "abdulRaHeeM452/Molecule-generator"  # your HF model repo


def load_models(repo_id: str = HF_REPO_ID):
    global _tokenizer, _meta_model, _device
    if _meta_model is not None:
        return  # already loaded

    print("Loading models from HF Hub...")
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {_device}")

    tok_path  = hf_hub_download(repo_id=repo_id, filename="smiles_tokenizer_v3.json")
    base_path = hf_hub_download(repo_id=repo_id, filename="grammar_engine_v3.pt")
    meta_path = hf_hub_download(repo_id=repo_id, filename="meta_engine_v3.pt")

    _tokenizer = SMILESTokenizer()
    _tokenizer.load(tok_path)

    vocab_size = len(_tokenizer.vocab)
    pad_idx    = _tokenizer.vocab['<PAD>']

    base_model = BaseGrammarTransformer(vocab_size=vocab_size).to(_device)
    base_model.load_state_dict(torch.load(base_path, map_location=_device))
    base_model.eval()

    encoder    = EnhancedContextEncoder(vocab_size=vocab_size).to(_device)
    meta       = RapidAdaptationEngine(base_model, encoder, pad_idx=pad_idx).to(_device)

    ckpt = torch.load(meta_path, map_location=_device)
    meta.context_encoder.load_state_dict(ckpt['encoder'])
    meta.lora.load_state_dict(ckpt['lora'])
    meta.eval()

    _meta_model = meta
    print(f"Models loaded. Vocab size: {vocab_size}")


def run_generation(support_smiles: list[str], n: int = 50, temperature: float = 0.8):
    """Public API — call this from app.py or FastAPI."""
    load_models()
    raw     = generate_raw(_meta_model, _tokenizer, support_smiles,
                           num_generate=n, temperature=temperature)
    metrics = evaluate_generation(raw, support_smiles)
    images  = []
    for smi in metrics['novel_smiles'][:20]:
        img = mol_to_pil(smi, size=(280, 280))
        props = get_drug_props(smi)
        if img:
            images.append({"smiles": smi, "image": img, "props": props})
    metrics['images'] = images
    return metrics
