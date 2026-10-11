import os
import json
import math
import random
import logging
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from torch.func import functional_call                      # torch >= 2.0
except ImportError:                                             # torch 1.12 - 1.13
    from torch.nn.utils.stateless import functional_call

from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import subgraph, to_dense_adj, degree
from sklearn.metrics import classification_report, roc_auc_score
from scipy.stats import chi2
import networkx as nx
import optuna
from empath import Empath
import spacy

# networkx / numpy compatibility shim
if not hasattr(np, "warnings"):
    np.warnings = warnings

TRAIN_DEVICE = torch.device("cuda:0")

# Significance level for the trigram likelihood-ratio test.
# Only trigrams with p < TRIGRAM_ALPHA are used as model features.
TRIGRAM_ALPHA = 0.01

# --------------------------------------------------------------------------- #
# GatorTron joint fine-tuning settings
# --------------------------------------------------------------------------- #
N_TRAINABLE_LAYERS = 4       # last K GatorTron encoder layers are trained jointly with the GNN
MAX_LEN = 2000               # max tokens per note (also controls cache size, see NoteCache)
ENC_CHUNK = 2                # notes per encoder mini-chunk (lower this if you hit CUDA OOM)
GNN_EPOCHS = 50              # joint epochs per Optuna trial
N_TRIALS = 7 #15
MAX_GRAD_NORM = 1.0          # gradient clipping (GNN + encoder)

USE_AMP = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
CACHE_DTYPE = torch.bfloat16 if USE_AMP else torch.float16   # dtype of cached frozen hidden states


def amp_ctx(device):
    """bf16 autocast for the trainable encoder layers (no-op on CPU / GPUs without bf16)."""
    return torch.autocast(
        device_type="cuda", dtype=torch.bfloat16,
        enabled=(USE_AMP and device.type == "cuda"),
    )


# --------------------------------------------------------------------------- #
# Setup Device & Logger
# --------------------------------------------------------------------------- #
def setup_device():
    if torch.cuda.is_available():
        return TRAIN_DEVICE
    return torch.device("cpu")


def setup_logger(output_dir):
    logger = logging.getLogger("joint_logger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(os.path.join(output_dir, "train.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #
def log_classification_metrics(logger, split, y_true, y_pred, y_prob=None):
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],                 # always 2 labels so target_names never mismatches
        target_names=["IP", "OP"],
        digits=4,
        zero_division=0,
    )
    logger.info(f"\n[{split}] Classification Report:\n{report}")

    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob[:, 1])
            logger.info(f"[{split}] ROC-AUC: {auc:.4f}")
        except Exception as e:
            logger.warning(f"[{split}] ROC-AUC computation failed: {e}")


# --------------------------------------------------------------------------- #
# Prediction saving
# --------------------------------------------------------------------------- #
def save_predictions(graph, y_true, y_pred, y_prob, split, trial_number, ip_f1, output_dir):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{split}_predictions_trial{trial_number}_ipf1_{ip_f1:.4f}_{ts}.csv"
    path = os.path.join(output_dir, fname)

    rows = []
    for i, node in enumerate(graph["nodes"]):
        rows.append({
            "patient_id": node["patient_id"],
            "true_label": "IP" if y_true[i] == 0 else "OP",
            "predicted_label": "IP" if y_pred[i] == 0 else "OP",
            "prob_IP": float(y_prob[i, 0]),
            "prob_OP": float(y_prob[i, 1]),
            "trial": trial_number,
            "ip_f1": ip_f1,
        })

    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Pre-processing text + trigrams
# --------------------------------------------------------------------------- #
nlp = spacy.load("en_core_web_sm")
nlp.max_length = max(nlp.max_length, 5_000_000)   # collated notes can be very long


def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = []
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.like_num:
            continue
        if tok.ent_type_ == "PERSON":
            continue
        if not tok.is_alpha:
            continue
        tokens.append(tok.lemma_)
    return tokens


def extract_trigrams(text):
    toks = preprocess_text(text)
    return [" ".join(toks[i:i + 3]) for i in range(len(toks) - 2)]


def trigram_vector(text, trigram_index):
    """
    Label-independent trigram count vector.

    trigram_index: dict {trigram -> column}, built from the *significant* trigrams only.
    The vector depends only on the note text, never on its label, so train and
    test features are built identically (no label leakage).
    """
    vec = np.zeros(len(trigram_index), dtype=np.float32)
    for tri, cnt in Counter(extract_trigrams(text)).items():
        j = trigram_index.get(tri)
        if j is not None:
            vec[j] = cnt
    return vec


# --------------------------------------------------------------------------- #
# Analyze trigrams (likelihood-ratio / G-test)
# --------------------------------------------------------------------------- #
def likelihood_ratio_test(trigram, ip_trigrams_dict, op_trigrams_dict, total_IP, total_OP):
    """
    Log-likelihood ratio (G^2) for one trigram, 2x2 table:
        IP: [a, total_IP - a]     OP: [b, total_OP - b]
    """
    a = ip_trigrams_dict.get(trigram, 0)
    b = op_trigrams_dict.get(trigram, 0)

    n11 = a
    n12 = total_IP - a
    n21 = b
    n22 = total_OP - b
    n = n11 + n12 + n21 + n22

    # Null: P(trigram | IP) == P(trigram | OP)
    p11_null = (n11 + n21) / n
    p12_null = (n12 + n22) / n

    # Alternative: separate rates per group
    p11_alt = n11 / (n11 + n12) if (n11 + n12) > 0 else 0
    p12_alt = n12 / (n11 + n12) if (n11 + n12) > 0 else 0
    p21_alt = n21 / (n21 + n22) if (n21 + n22) > 0 else 0
    p22_alt = n22 / (n21 + n22) if (n21 + n22) > 0 else 0

    def safe_log(x):
        return np.log(x) if x > 0 else 0

    ll_null = (
        n11 * safe_log(p11_null) + n12 * safe_log(p12_null) +
        n21 * safe_log(p11_null) + n22 * safe_log(p12_null)
    )
    ll_alt = (
        n11 * safe_log(p11_alt) + n12 * safe_log(p12_alt) +
        n21 * safe_log(p21_alt) + n22 * safe_log(p22_alt)
    )
    return 2 * (ll_alt - ll_null)


def analyze_trigrams_from_csv(ip_csv_path, op_csv_path, alpha=TRIGRAM_ALPHA):
    """
    Returns:
        significant_trigrams : sorted list of all trigrams with p < alpha (IP- or OP-enriched)
        ip_significant_set   : significant trigrams with a higher relative frequency in IP
        op_significant_set   : significant trigrams with a higher relative frequency in OP
        results_df           : full per-trigram results table (sorted by p-value)
    """
    ip_df = pd.read_csv(ip_csv_path)
    op_df = pd.read_csv(op_csv_path)

    ip_trigrams_dict = dict(zip(ip_df['Trigram'], ip_df['IP_Count']))
    op_trigrams_dict = dict(zip(op_df['Trigram'], op_df['OP_Count']))

    total_IP = ip_df['IP_Count'].sum()
    total_OP = op_df['OP_Count'].sum()

    all_trigrams = set(ip_trigrams_dict) | set(op_trigrams_dict)

    significant_trigrams = []
    ip_significant_set, op_significant_set = set(), set()
    results_data = []

    for trigram in sorted(all_trigrams):
        a = ip_trigrams_dict.get(trigram, 0)
        b = op_trigrams_dict.get(trigram, 0)

        lr_statistic = likelihood_ratio_test(
            trigram, ip_trigrams_dict, op_trigrams_dict, total_IP, total_OP
        )
        p_value = float(chi2.sf(lr_statistic, 1))      # sf() is exact for tiny p; 1 - cdf() underflows to 0
        is_significant = p_value < alpha

        # Compare *relative* frequencies; raw counts are biased toward the larger class.
        ip_rate = a / total_IP if total_IP > 0 else 0.0
        op_rate = b / total_OP if total_OP > 0 else 0.0
        if ip_rate > op_rate:
            enriched_in = "IP"
        elif op_rate > ip_rate:
            enriched_in = "OP"
        else:
            enriched_in = "both"

        results_data.append({
            'Trigram': trigram,
            'IP_Count': a,
            'OP_Count': b,
            'IP_Rate': ip_rate,
            'OP_Rate': op_rate,
            'Enriched_In': enriched_in,
            'Likelihood_Ratio': lr_statistic,
            'P_Value': p_value,
            'Statistically_Significant': is_significant,
        })

        if is_significant:
            significant_trigrams.append(trigram)
            if enriched_in in ("IP", "both"):
                ip_significant_set.add(trigram)
            if enriched_in in ("OP", "both"):
                op_significant_set.add(trigram)

    results_df = pd.DataFrame(results_data).sort_values('P_Value').reset_index(drop=True)
    return sorted(significant_trigrams), ip_significant_set, op_significant_set, results_df


# --------------------------------------------------------------------------- #
# GatorTron encoder with the last K layers trainable
# --------------------------------------------------------------------------- #
class GatorTronEncoder(nn.Module):
    """
    GatorTron where only the last `n_trainable_layers` transformer layers are trainable.

    Because the first (n_layers - K) layers are frozen *and* run in eval mode, their output
    is a fixed function of the note text. We therefore run them ONCE per note and cache the
    result (see NoteCache). Training/evaluation then only re-runs the K trainable layers
    (+ the frozen final LayerNorm) on top of the cached hidden states, which is mathematically
    identical to running the whole network.
    """

    def __init__(self, model_id, n_trainable_layers=4):
        super().__init__()
        # Recent transformers versions load weights in the checkpoint dtype (fp16 for GatorTron)
        # instead of fp32. Fine-tuning needs fp32 master weights (AdamW on fp16 weights is unstable,
        # and the cached fp32 hidden states would not match fp16 LayerNorm weights), so upcast.
        self.model = AutoModel.from_pretrained(model_id).float()
        self.hidden_size = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers
        self.k = n_trainable_layers

        for p in self.model.parameters():
            p.requires_grad = False
        for layer in self.model.encoder.layer[-self.k:]:
            for p in layer.parameters():
                p.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def snapshot_trainable(self):
        """CPU copy of the trainable weights (used to reset every Optuna trial + to save best model)."""
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable(self, state):
        params = dict(self.named_parameters())
        with torch.no_grad():
            for n, v in state.items():
                params[n].copy_(v)

    # Original behaviour: CLS embedding of the full network
    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

    @torch.no_grad()
    def frozen_forward(self, input_ids, attention_mask):
        """
        One full forward pass. Returns
          cls    : [B, H]     CLS embedding with the *pretrained* weights (initial embedding)
          hidden : [B, L, H]  input to the first trainable layer (output of the last frozen layer)
        hidden_states[i] is the input of layer i, so index (n_layers - K) is what we need.
        """
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return out.last_hidden_state[:, 0, :], out.hidden_states[self.n_layers - self.k]

    def forward_trainable(self, hidden, attention_mask):
        """
        hidden: [B, L, H] float32 cached input of the first trainable layer
        attention_mask: [B, L] (1 = token, 0 = pad)
        returns CLS embedding [B, H] (gradients flow into the K trainable layers only)
        """
        ext = (1.0 - attention_mask[:, None, None, :].to(hidden.dtype)) * torch.finfo(hidden.dtype).min
        for layer in self.model.encoder.layer[-self.k:]:
            out = layer(hidden, attention_mask=ext)
            hidden = out[0] if isinstance(out, (tuple, list)) else out
        final_ln = getattr(self.model.encoder, "ln", None)   # final (frozen) LayerNorm of Megatron-BERT
        if final_ln is not None:
            hidden = final_ln(hidden)
        return hidden[:, 0, :]


class NoteCache:
    """
    CPU cache of the frozen-stack hidden states, one [len_i, H] tensor per note (no padding stored).
    Memory ~= total_tokens * hidden_size * 2 bytes (bf16/fp16). Lower MAX_LEN if this gets too big.
    Notes are grouped into length-sorted chunks so padding inside a chunk is minimal.
    """

    def __init__(self, hidden_list, chunk_size):
        self.hidden = hidden_list
        self.n = len(hidden_list)
        lengths = [h.size(0) for h in hidden_list]
        order = sorted(range(self.n), key=lambda i: lengths[i])
        self.chunks = [order[i:i + chunk_size] for i in range(0, self.n, chunk_size)]
        self.chunk_idx = [torch.tensor(c, dtype=torch.long) for c in self.chunks]
        self.size_gb = sum(h.numel() * h.element_size() for h in hidden_list) / 1e9

    def batch(self, chunk_id, device):
        seqs = [self.hidden[i] for i in self.chunks[chunk_id]]
        max_len = max(s.size(0) for s in seqs)
        hidden = torch.zeros(len(seqs), max_len, seqs[0].size(1), dtype=seqs[0].dtype)
        mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for j, s in enumerate(seqs):
            hidden[j, :s.size(0)] = s
            mask[j, :s.size(0)] = 1
        return hidden.to(device).float(), mask.to(device)

    def single(self, i, device):
        h = self.hidden[i].unsqueeze(0).to(device).float()
        return h, torch.ones(1, h.size(1), dtype=torch.long, device=device)


def build_note_cache(encoder, tokenizer, texts, device, logger=None, name=""):
    """
    One pass over all notes with the pretrained network.
    Returns (NoteCache, initial_cls_embeddings[N, H]).
    The initial CLS embeddings equal the old precomputed GatorTron features; they are only used
    to build data.x for meta-set selection. During training they are replaced by live embeddings.
    """
    encoder.eval()
    hidden_list, cls_list = [], []
    with torch.no_grad():
        for t in texts:
            inp = tokenizer(t, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
            cls, h = encoder.frozen_forward(inp["input_ids"], inp["attention_mask"])
            h = h.squeeze(0)
            if CACHE_DTYPE == torch.float16 and not torch.isfinite(h.to(CACHE_DTYPE)).all():
                warnings.warn("fp16 overflow while caching hidden states; use a bf16-capable GPU or fp32 cache")
            cls_list.append(cls.squeeze(0).float().cpu().numpy())
            hidden_list.append(h.to(CACHE_DTYPE).cpu())

    cache = NoteCache(hidden_list, ENC_CHUNK)
    if logger:
        logger.info(f"[{name}] cached {cache.n} notes, {cache.size_gb:.2f} GB in CPU RAM ({CACHE_DTYPE})")
    return cache, np.stack(cls_list)


def sanity_check_partial_forward(encoder, cache, init_cls, device, logger, n=3):
    """Checks cached-hidden -> trainable layers reproduces the full network's CLS embedding."""
    encoder.eval()
    with torch.no_grad():
        for i in range(min(n, cache.n)):
            h, m = cache.single(i, device)
            emb = encoder.forward_trainable(h, m).squeeze(0).float().cpu()
            ref = torch.tensor(init_cls[i])
            cos = F.cosine_similarity(emb, ref, dim=0).item()
            logger.info(f"[sanity] note {i}: cosine(partial, full CLS) = {cos:.5f}")
            if cos < 0.99:
                logger.warning(
                    "Partial forward does not reproduce the full forward. The layer call signature / "
                    "attention-mask format may differ in your transformers version - check forward_trainable()."
                )


def embed_nograd(encoder, cache, device):
    """Live CLS embeddings for all notes with the *current* trainable weights (no graph kept)."""
    H = torch.zeros(cache.n, encoder.hidden_size, device=device)
    with torch.no_grad(), amp_ctx(device):
        for cid in range(len(cache.chunks)):
            hid, mask = cache.batch(cid, device)
            idx = cache.chunk_idx[cid].to(device)
            H[idx] = encoder.forward_trainable(hid, mask).float()
    return H


def encoder_backward(encoder, cache, grad_H, device):
    """
    Second half of the two-stage backward (same idea as GradCache): grad_H = dLoss/dEmbedding was
    obtained from the GNN. Re-run each chunk WITH autograd and push its slice of grad_H into the
    trainable GatorTron layers. Chunks with an all-zero gradient (e.g. isolated nodes) are skipped.
    """
    for cid in range(len(cache.chunks)):
        idx = cache.chunk_idx[cid].to(device)
        g = grad_H[idx]
        if not torch.any(g != 0):
            continue
        hid, mask = cache.batch(cid, device)
        with amp_ctx(device):
            emb = encoder.forward_trainable(hid, mask)
        emb.backward(g.to(emb.dtype))


# --------------------------------------------------------------------------- #
# GNN
# --------------------------------------------------------------------------- #
class MetaGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.lin(x))
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)


# --------------------------------------------------------------------------- #
# Joint meta-learning trainer: GNN + last-K GatorTron layers
# --------------------------------------------------------------------------- #
def train_joint_meta_gnn(
    gnn, encoder, cache, data, optimizer, criterion, gt_dim,
    meta_lr=1e-3, epochs=10, logger=None, max_grad_norm=MAX_GRAD_NORM,
):
    """
    Per epoch:
      1. live embeddings H for every node from the current trainable GatorTron layers (no graph)
      2. meta step: virtual SGD step on the GNN with the per-sample weights, meta-loss on the clean
         meta set, meta-gradient -> update the sample weights.
         (The encoder is treated as fixed inside the virtual step; differentiating through the
         transformer layers twice would be far too expensive.)
      3. real step: weighted loss through GNN -> grads for GNN params and dLoss/dH,
         then dLoss/dH is back-propagated chunk-by-chunk into the trainable GatorTron layers.
         One AdamW step updates both (param group 0 = GNN, group 1 = GatorTron).

    The encoder runs in eval mode (no dropout) so passes 1 and 3 are identical.
    criterion must return per-sample losses (nn.NLLLoss(reduction="none")).
    """
    device = data.x.device
    ei = data.edge_index
    x_static = data.x[:, gt_dim:]                       # Empath | trigrams | reasoning (fixed)
    train_idx = (~data.meta_mask).nonzero(as_tuple=True)[0]
    meta_idx = data.meta_mask.nonzero(as_tuple=True)[0]
    y_train, y_meta = data.y[train_idx], data.y[meta_idx]
    all_params = [p for g in optimizer.param_groups for p in g["params"]]

    weights = nn.Parameter(torch.ones(len(train_idx), device=device) / len(train_idx))
    meta_opt = torch.optim.SGD([weights], lr=meta_lr)

    for epoch in range(epochs):
        gnn.train()
        encoder.eval()
        inner_lr = optimizer.param_groups[0]["lr"]      # GNN learning rate

        # 1) live embeddings
        H = embed_nograd(encoder, cache, device)

        # 2) meta step (GNN-only virtual update)
        x_in = torch.cat([H, x_static], dim=1)
        params = dict(gnn.named_parameters())
        out = functional_call(gnn, params, (Data(x=x_in, edge_index=ei),))
        losses = criterion(out[train_idx], y_train)
        weighted_loss = torch.sum(weights * losses)

        grads = torch.autograd.grad(
            weighted_loss, list(params.values()), create_graph=True, allow_unused=True
        )
        fast_weights = {
            name: p - inner_lr * (g if g is not None else 0.0)
            for (name, p), g in zip(params.items(), grads)
        }
        meta_out = functional_call(gnn, fast_weights, (Data(x=x_in, edge_index=ei),))
        meta_loss = criterion(meta_out[meta_idx], y_meta).mean()

        w_grad = torch.autograd.grad(meta_loss, weights)[0]
        meta_opt.zero_grad()
        weights.grad = w_grad.detach()
        meta_opt.step()

        with torch.no_grad():
            w = torch.clamp(weights, min=0)
            s = w.sum()
            weights.copy_(w / s if s > 0 else torch.ones_like(w) / len(w))

        # 3) real joint step
        H_leaf = H.clone().requires_grad_(True)
        out = gnn(Data(x=torch.cat([H_leaf, x_static], dim=1), edge_index=ei))
        final_losses = criterion(out[train_idx], y_train)
        final_weighted_loss = torch.sum(weights.detach() * final_losses)

        optimizer.zero_grad()
        final_weighted_loss.backward()                  # GNN grads + H_leaf.grad
        encoder_backward(encoder, cache, H_leaf.grad, device)   # GatorTron grads
        torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
        optimizer.step()

        if logger and epoch % 2 == 0:
            logger.info(
                f"[Joint Epoch {epoch}] "
                f"Train Loss={weighted_loss.item():.4f} "
                f"Meta Loss={meta_loss.item():.4f}"
            )


@torch.no_grad()
def predict_log_probs(gnn, encoder, cache, data, gt_dim, device):
    """Embeddings from the (fine-tuned) encoder -> GNN -> log-probs for every node of a graph."""
    gnn.eval()
    encoder.eval()
    H = embed_nograd(encoder, cache, device)
    x = torch.cat([H, data.x[:, gt_dim:]], dim=1)
    return gnn(Data(x=x, edge_index=data.edge_index))


# --------------------------------------------------------------------------- #
# Graph -> PyG Data
# --------------------------------------------------------------------------- #
def load_graph(json_path):
    with open(json_path) as f:
        return json.load(f)


def build_data_object(graph, gatortron_embs, empath, empath_cats, trigram_list, reasoning_json, device):
    """
    gatortron_embs: *initial* (pretrained) CLS embeddings. They fill the first GT_DIM columns of x
                    (used for meta-set selection); during training those columns are replaced by
                    live embeddings from the fine-tuned encoder.
    trigram_list:   the *significant* trigrams only (feature columns).
    Features = [GatorTron CLS | Empath | significant-trigram counts | reasoning embedding]
    """
    reasoning_map = {}
    reason_dim = 384
    if reasoning_json:
        with open(reasoning_json) as f:
            for e in json.load(f):
                if "patient_id" in e and "embedding" in e:
                    reasoning_map[e["patient_id"]] = np.array(e["embedding"], dtype=np.float32)
        if reasoning_map:
            reason_dim = len(next(iter(reasoning_map.values())))

    trigram_index = {t: j for j, t in enumerate(trigram_list)}
    node_map = {n["patient_id"]: i for i, n in enumerate(graph["nodes"])}
    X, Y = [], []

    for i, n in enumerate(graph["nodes"]):
        text = n["collated_notes"]
        label = n["label"].strip().upper()
        if label not in ("IP", "OP"):
            raise ValueError(f"Unexpected label {n['label']!r} for patient {n['patient_id']}")
        y = 0 if label == "IP" else 1

        gt = gatortron_embs[i]

        emp = empath.analyze(text, normalize=True) or {}      # analyze() can return None
        emp_vec = np.array([emp.get(c, 0.0) for c in empath_cats], dtype=np.float32)

        tri_vec = trigram_vector(text, trigram_index)         # label-independent, significant only

        rsn = reasoning_map.get(n["patient_id"], np.zeros(reason_dim, dtype=np.float32))

        X.append(np.concatenate([gt, emp_vec, tri_vec, rsn]))
        Y.append(y)

    edges = [[node_map[e["source"]], node_map[e["target"]]] for e in graph["edges"]]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(
        x=torch.tensor(np.stack(X).astype(np.float32)).to(device),
        y=torch.tensor(Y, dtype=torch.long).to(device),
        edge_index=edge_index.to(device),
    )


# --------------------------------------------------------------------------- #
# TreeNorm helpers
# --------------------------------------------------------------------------- #
def induced_subgraph_data(data, nodes):
    """Induced subgraph (CPU) on `nodes`, using PyG's subgraph() instead of hand-rolled masks."""
    subset = torch.unique(torch.as_tensor(nodes, dtype=torch.long, device=data.edge_index.device))
    sub_ei, _ = subgraph(subset, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    return Data(
        x=data.x[subset.to(data.x.device)].detach().clone().cpu(),
        edge_index=sub_ei.detach().cpu(),
    )


def compute_dynamic_L_w(data, output_dir=None, L_min=1, L_max=4, alpha=1.0):
    """
    Per-node depth L_v and weights w_v. JSON files are only written when output_dir is given
    (previously they were rewritten on every fitness call with sub-graph values).
    """
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    d_min = deg.min().item()
    d_max = deg.max().item()
    eps = 1e-5
    d_norm = ((deg - d_min) / (d_max - d_min + eps)).tolist()

    L_dict, w_dict = {}, {}
    for v, d_v in enumerate(d_norm):
        L_v = int(L_max - (L_max - L_min) * d_v)
        lambda_v = math.exp(-alpha * d_v)
        L_dict[v] = L_v
        w_dict[v] = [lambda_v ** l for l in range(1, L_v)]       # w(0) skipped

    if output_dir is not None:
        with open(os.path.join(output_dir, "L_dict.json"), "w") as f:
            json.dump({str(k): int(v) for k, v in L_dict.items()}, f, indent=2)
        with open(os.path.join(output_dir, "w_dict.json"), "w") as f:
            json.dump({str(k): [float(w) for w in v] for k, v in w_dict.items()}, f, indent=2)

    return L_dict, w_dict, int(max(L_dict.values()))


def tree_norm_dynamic(data, output_dir=None, L_min=1, L_max=4, alpha=1.0):
    """Dynamic TreeNorm with node-wise depth L_v and decay weights w_v."""
    assert L_max >= L_min >= 1
    L_dict, w_dict, L_global = compute_dynamic_L_w(data, output_dir, L_min, L_max, alpha)

    N = data.num_nodes
    if data.edge_index.numel() == 0:
        A = torch.zeros(N, N, device=data.x.device)
    else:
        A = to_dense_adj(data.edge_index, max_num_nodes=N)[0]       # [N, N]

    z = [torch.norm(data.x, p=2, dim=1)]                            # [N]
    for ell in range(1, L_global):
        z.append(torch.matmul(A, z[ell - 1]))
    Z = torch.stack(z)                                              # [L_global, N]

    # b_v = z_0[v] + sum_{ell=1}^{L_v-1} w_v[ell-1] * z_ell[v]
    C = [[0.0] * L_global for _ in range(N)]
    for v in range(N):
        C[v][0] = 1.0
        for ell in range(1, L_dict[v]):
            C[v][ell] = w_dict[v][ell - 1]
    C = torch.tensor(C, dtype=Z.dtype, device=Z.device)
    b = (C * Z.t()).sum(dim=1)
    return b.abs().sum().item()


# --------------------------------------------------------------------------- #
# Meta dataset creation (GA sampling)
# --------------------------------------------------------------------------- #
def create_meta_dataset(data, output_dir, meta_ratio=0.1, seed=None, logger=None):
    """
    GA-based sampling of a class-balanced meta set.
    Sets data.meta_mask and returns (data, selected_indices).
    """
    log = logger.info if logger is not None else (lambda *a, **k: None)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    log(f"Seed = {seed}")
    device = data.x.device

    N = data.y.size(0)
    labels = data.y.cpu().numpy()
    idx_IP = np.where(labels == 0)[0].tolist()      # minority class (IP)
    idx_OP = np.where(labels == 1)[0].tolist()      # majority class (OP)

    n_each = int(meta_ratio * N) // 2
    max_each = min(len(idx_IP), len(idx_OP))
    if n_each > max_each:
        log(f"Requested {n_each} per class but minority class only has {max_each}; shrinking meta set.")
        n_each = max_each
    if n_each < 2:
        raise ValueError(f"Meta set too small (n_each={n_each}); increase meta_ratio or data size.")

    E = data.x.cpu().numpy()
    V_total = np.var(E, axis=0).sum()

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(data.edge_index.t().tolist())
    communities = nx.community.greedy_modularity_communities(G)
    comm_map = {node: cid for cid, comm in enumerate(communities) for node in comm}
    n_communities = len(communities)
    clustering_coeff_G = nx.average_clustering(G)
    with np.errstate(all="ignore"):
        assortativity_G = nx.degree_assortativity_coefficient(G) if G.number_of_edges() > 0 else 0.0

    ei = data.edge_index.cpu().numpy()
    deg = np.zeros(N, dtype=int)
    np.add.at(deg, ei[0], 1)
    np.add.at(deg, ei[1], 1)
    mean_deg = deg.mean()

    # GA parameters
    pop_size = 50
    generations = 100
    crossover_rate = 0.8745703676281257
    mutation_rate = 0.21873583752075254
    log(f"GA parameters: population size = {pop_size}, #generations = {generations}, "
        f"crossover_rate = {crossover_rate}, and mutation_rate = {mutation_rate}.")

    population = [
        (random.sample(idx_IP, n_each), random.sample(idx_OP, n_each))
        for _ in range(pop_size)
    ]
    log("population initialized...")

    def safe_ratio(num, den):
        # ratios against assortativity_G / clustering_coeff_G can be 0/0, NaN or inf
        if not (np.isfinite(num) and np.isfinite(den)) or abs(den) < 1e-12:
            return 0.0
        return float(num / den)

    def compute_structural_metrics(sub_nodes):
        sg = G.subgraph(sub_nodes)
        cc = nx.average_clustering(sg)
        with np.errstate(all="ignore"):
            # assortativity is NaN on sparse/regular induced subgraphs (common for a 10% sample)
            assort = nx.degree_assortativity_coefficient(sg) if sg.number_of_edges() > 0 else 0.0
        return cc, (assort if np.isfinite(assort) else 0.0)

    tree_norm_G = tree_norm_dynamic(data, output_dir, L_min=1, L_max=4, alpha=1.0)  # only call that writes JSON

    def fitness(individual):
        sel_IP, sel_OP = individual
        sel = sel_IP + sel_OP

        # Structural diversity
        f_deg = deg[sel].mean() / (mean_deg + 1e-9)
        f_comm = len({comm_map[i] for i in sel}) / n_communities
        cc, assort = compute_structural_metrics(sel)
        f_struct = f_deg + f_comm + safe_ratio(cc, clustering_coeff_G) + safe_ratio(assort, assortativity_G)

        # Semantic variance
        f_semantic = np.var(E[sel], axis=0).sum() / (V_total + 1e-9)

        # TreeNorm similarity
        sampled_graph = induced_subgraph_data(data, sel)
        tn_sub = tree_norm_dynamic(sampled_graph, None, L_min=1, L_max=4, alpha=1.0)
        normalized_treenorm = safe_ratio(math.log1p(tn_sub), math.log1p(tree_norm_G))

        return f_struct + f_semantic + normalized_treenorm

    def crossover_pair(a, b, pool):
        cut = random.randint(1, n_each - 1)
        c1 = list(dict.fromkeys(a[:cut] + b[cut:]))
        c2 = list(dict.fromkeys(b[:cut] + a[cut:]))
        for c in (c1, c2):
            seen = set(c)
            while len(c) < n_each:
                cand = random.choice(pool)
                if cand not in seen:
                    c.append(cand)
                    seen.add(cand)
        return c1[:n_each], c2[:n_each]

    def mutate(sel_list, full_list):
        if random.random() < mutation_rate:
            i = random.randrange(len(sel_list))
            in_sel = set(sel_list)
            candidates = [x for x in full_list if x not in in_sel]
            if candidates:
                sel_list[i] = random.choice(candidates)
        return sel_list

    # Main GA loop
    for gen in range(generations):
        fitness_vals = np.array([fitness(ind) for ind in population], dtype=np.float64)
        # NaN / negative fitness would make the roulette probabilities invalid
        fitness_vals = np.clip(np.nan_to_num(fitness_vals, nan=0.0), 1e-9, None)
        probs = fitness_vals / fitness_vals.sum()

        if gen % 10 == 0:
            log(f"[GA gen {gen}] best fitness = {fitness_vals.max():.4f}, mean = {fitness_vals.mean():.4f}")

        new_pop = []
        while len(new_pop) < pop_size:
            parents = np.random.choice(pop_size, size=2, p=probs, replace=False)
            parent1, parent2 = population[parents[0]], population[parents[1]]

            if random.random() < crossover_rate:
                child1_IP, child2_IP = crossover_pair(parent1[0], parent2[0], idx_IP)
                child1_OP, child2_OP = crossover_pair(parent1[1], parent2[1], idx_OP)
            else:
                child1_IP, child1_OP = parent1[0][:], parent1[1][:]
                child2_IP, child2_OP = parent2[0][:], parent2[1][:]

            child1_IP = mutate(child1_IP, idx_IP)
            child2_IP = mutate(child2_IP, idx_IP)
            child1_OP = mutate(child1_OP, idx_OP)
            child2_OP = mutate(child2_OP, idx_OP)

            new_pop.extend([(child1_IP, child1_OP), (child2_IP, child2_OP)])

        population = new_pop[:pop_size]

    best_ind = max(population, key=lambda ind: np.nan_to_num(fitness(ind), nan=-np.inf))
    sel_IP, sel_OP = best_ind
    selected = sel_IP + sel_OP

    meta_mask = torch.zeros(N, dtype=torch.bool, device=device)
    meta_mask[selected] = True
    data.meta_mask = meta_mask

    log(f"Created meta-dataset with {len(selected)} samples")
    return data, selected


# --------------------------------------------------------------------------- #
# Optuna objective
# --------------------------------------------------------------------------- #
def objective_factory(
    train_data, test_data, gatortron, train_cache, test_cache, gt_dim,
    device, output_dir, logger, train_graph, test_graph,
):
    best_ip_f1 = {"val": 0.0}
    init_state = gatortron.snapshot_trainable()          # pretrained weights of the K trainable layers

    def objective(trial):
        hidden_dim = trial.suggest_int("hidden_dim", 32, 256)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)               # GNN learning rate
        enc_lr = trial.suggest_float("enc_lr", 1e-6, 1e-4, log=True)       # GatorTron learning rate
        meta_lr = trial.suggest_float("meta_lr", 1e-4, 5e-3, log=True)
        torch.manual_seed(42 + trial.number)

        # every trial must start from the PRETRAINED GatorTron weights, not the previous trial's
        gatortron.load_trainable(init_state)
        gnn = MetaGNN(train_data.x.size(1), hidden_dim, 2).to(device)

        # group 0 = GNN (its lr is also the inner lr of the meta step), group 1 = GatorTron.
        # They need very different learning rates: 5e-3 would wreck pretrained transformer layers.
        optimizer = optim.AdamW([
            {"params": list(gnn.parameters()), "lr": lr},
            {"params": gatortron.trainable_parameters(), "lr": enc_lr},
        ])
        criterion = nn.NLLLoss(reduction="none")

        train_joint_meta_gnn(
            gnn=gnn,
            encoder=gatortron,
            cache=train_cache,
            data=train_data,
            optimizer=optimizer,
            criterion=criterion,
            gt_dim=gt_dim,
            meta_lr=meta_lr,
            epochs=GNN_EPOCHS,
            logger=logger,
        )

        # Evaluation: embeddings come from the fine-tuned encoder, not from a precomputed table
        train_out = predict_log_probs(gnn, gatortron, train_cache, train_data, gt_dim, device)
        train_preds = train_out.argmax(dim=1)
        train_probs = torch.exp(train_out)

        log_classification_metrics(
            logger,
            split=f"Trial {trial.number} | TRAIN",
            y_true=train_data.y.cpu().numpy(),
            y_pred=train_preds.cpu().numpy(),
            y_prob=train_probs.cpu().numpy(),
        )

        test_out = predict_log_probs(gnn, gatortron, test_cache, test_data, gt_dim, device)
        test_preds = test_out.argmax(dim=1)
        test_probs = torch.exp(test_out)

        log_classification_metrics(
            logger,
            split=f"Trial {trial.number} | TEST",
            y_true=test_data.y.cpu().numpy(),
            y_pred=test_preds.cpu().numpy(),
            y_prob=test_probs.cpu().numpy(),
        )

        report = classification_report(
            test_data.y.cpu().numpy(),
            test_preds.cpu().numpy(),
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        )
        ip_f1 = report["0"]["f1-score"]

        if ip_f1 > best_ip_f1["val"]:
            logger.info(
                f"New Best Model Found | Trial {trial.number} | "
                f"IP-F1: {ip_f1:.4f} (prev {best_ip_f1['val']:.4f})"
            )
            best_ip_f1["val"] = ip_f1

            # Save GNN + only the fine-tuned GatorTron layers (~4 layers, not the whole 345M model).
            # To restore: load pretrained GatorTron, then load_trainable(ckpt["gatortron_trainable"]).
            torch.save(
                {
                    "gnn": gnn.state_dict(),
                    "gatortron_trainable": gatortron.snapshot_trainable(),
                    "n_trainable_layers": N_TRAINABLE_LAYERS,
                    "in_dim": train_data.x.size(1),
                    "hidden_dim": hidden_dim,
                    "trial": trial.number,
                    "params": trial.params,
                    "ip_f1": ip_f1,
                },
                os.path.join(output_dir, "best_model.pt"),
            )

            train_pred_path = save_predictions(
                graph=train_graph,
                y_true=train_data.y.cpu().numpy(),
                y_pred=train_preds.cpu().numpy(),
                y_prob=train_probs.cpu().numpy(),
                split="train",
                trial_number=trial.number,
                ip_f1=ip_f1,
                output_dir=output_dir,
            )
            test_pred_path = save_predictions(
                graph=test_graph,
                y_true=test_data.y.cpu().numpy(),
                y_pred=test_preds.cpu().numpy(),
                y_prob=test_probs.cpu().numpy(),
                split="test",
                trial_number=trial.number,
                ip_f1=ip_f1,
                output_dir=output_dir,
            )
            logger.info(f"Saved train predictions -> {train_pred_path}")
            logger.info(f"Saved test predictions  -> {test_pred_path}")

        del optimizer, gnn
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return ip_f1

    return objective


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    device = setup_device()
    outdir = f"joint_run_ga_wost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(outdir, exist_ok=True)
    logger = setup_logger(outdir)

    # Load graphs
    train_graph = load_graph("same_graph_train_bothmasked.json")
    test_graph = load_graph("same_graph_test_bothmasked.json")

    # GatorTron: last N_TRAINABLE_LAYERS layers trainable, rest frozen
    tokenizer = AutoTokenizer.from_pretrained("UFNLP/gatortron-base-2k")
    gatortron = GatorTronEncoder("UFNLP/gatortron-base-2k", n_trainable_layers=N_TRAINABLE_LAYERS).to(device)
    gatortron.eval()      # stays in eval mode (no dropout) so the two-stage backward is exact
    n_train_params = sum(p.numel() for p in gatortron.trainable_parameters())
    logger.info(f"GatorTron: training last {N_TRAINABLE_LAYERS}/{gatortron.n_layers} layers "
                f"({n_train_params / 1e6:.1f}M params), AMP bf16={USE_AMP}")

    # One pass with the pretrained network: cache the frozen-stack output + initial CLS embeddings
    train_cache, train_gt = build_note_cache(
        gatortron, tokenizer, [n["collated_notes"] for n in train_graph["nodes"]], device, logger, "train")
    test_cache, test_gt = build_note_cache(
        gatortron, tokenizer, [n["collated_notes"] for n in test_graph["nodes"]], device, logger, "test")
    sanity_check_partial_forward(gatortron, train_cache, train_gt, device, logger)

    # Empath
    empath = Empath()
    empath_cats = sorted(empath.cats)

    # ------------------------------------------------------------------ #
    # Trigrams: keep ONLY the statistically significant ones as features
    # ------------------------------------------------------------------ #
    ip_csv_path = 'ip_specific_trigrams_masked_train_new_bothmasked.csv'
    op_csv_path = 'op_specific_trigrams_masked_train_new_bothmasked.csv'

    # logger.info("Analyzing trigrams (likelihood-ratio test)...")
    # significant_trigrams, ip_sig_set, op_sig_set, results_df = analyze_trigrams_from_csv(
    #     ip_csv_path, op_csv_path, alpha=TRIGRAM_ALPHA
    # )

    # trigram_list = sorted(set(significant_trigrams))     # feature columns (significant only)
    # if not trigram_list:
    #     raise RuntimeError(
    #         f"No trigram passed the significance test at alpha={TRIGRAM_ALPHA}; "
    #         "check the CSVs or relax alpha."
    #     )

    # results_df.to_csv(os.path.join(outdir, "trigram_lrt_results.csv"), index=False)
    # with open(os.path.join(outdir, "significant_trigrams.json"), "w") as f:
    #     json.dump(trigram_list, f, indent=2)

    # logger.info(
    #     f"Significant trigrams used as features: {len(trigram_list)} of {len(results_df)} "
    #     f"(alpha={TRIGRAM_ALPHA}) | IP-enriched: {len(ip_sig_set)}, OP-enriched: {len(op_sig_set)}"
    # )

    ip_trigrams = set(pd.read_csv("ip_specific_trigrams_masked_train_new_bothmasked.csv")["Trigram"])
    op_trigrams = set(pd.read_csv("op_specific_trigrams_masked_train_new_bothmasked.csv")["Trigram"])
    trigram_list = sorted(ip_trigrams | op_trigrams)

    GT_DIM = gatortron.hidden_size
    EMPATH_DIM = len(empath_cats)
    TRIGRAM_DIM = len(trigram_list)



    # Build data (first GT_DIM columns = initial pretrained CLS embeddings)
    train_data = build_data_object(
        train_graph, train_gt, empath, empath_cats, trigram_list,
        "or_train_zsl_gptoss_embeddings.json", device,
    )

    train_data, meta_selected_idx = create_meta_dataset(
        train_data,
        output_dir=outdir,
        meta_ratio=0.1,
        seed=42,
        logger=logger,
    )

    test_data = build_data_object(
        test_graph, test_gt, empath, empath_cats, trigram_list,
        "or_test_zsl_gptoss_embeddings.json", device,
    )

    FEATURE_RANGES = {
        "gatortron": (0, GT_DIM),
        "empath": (GT_DIM, GT_DIM + EMPATH_DIM),
        "trigrams": (GT_DIM + EMPATH_DIM, GT_DIM + EMPATH_DIM + TRIGRAM_DIM),
        "reasoning": (GT_DIM + EMPATH_DIM + TRIGRAM_DIM, train_data.x.size(1)),
    }
    logger.info(f"Feature ranges: {FEATURE_RANGES}")

    # Sanity check: how many notes actually hit at least one significant trigram?
    tri_lo, tri_hi = FEATURE_RANGES["trigrams"]
    for name, d in (("train", train_data), ("test", test_data)):
        cov = (d.x[:, tri_lo:tri_hi].sum(dim=1) > 0).float().mean().item()
        logger.info(f"{name}: {cov:.1%} of notes contain >= 1 significant trigram")

    logger.info("========== Dataset Statistics ==========")
    logger.info(f"Train Nodes: {train_data.num_nodes}")
    logger.info(f"Train Edges: {train_data.num_edges}")
    logger.info(f"Train IP: {(train_data.y == 0).sum().item()}, OP: {(train_data.y == 1).sum().item()}")
    logger.info(f"Test Nodes: {test_data.num_nodes}")
    logger.info(f"Test Edges: {test_data.num_edges}")
    logger.info(f"Test IP: {(test_data.y == 0).sum().item()}, OP: {(test_data.y == 1).sum().item()}")
    logger.info("========================================")

    # Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        objective_factory(
            train_data, test_data, gatortron, train_cache, test_cache, GT_DIM,
            device, outdir, logger, train_graph, test_graph,
        ),
        n_trials=N_TRIALS,
    )

    logger.info(f"Best trial: {study.best_trial.params}")


if __name__ == "__main__":
    main()
