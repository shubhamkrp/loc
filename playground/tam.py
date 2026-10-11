import os
import json
import random
import logging
from datetime import datetime
from copy import deepcopy


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import subgraph
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.manifold import TSNE
import optuna
import matplotlib.pyplot as plt


from captum.attr import IntegratedGradients
TRAIN_DEVICE = torch.device("cuda:0")
ATTR_DEVICE = torch.device("cuda:1")




from empath import Empath
import spacy
import re
import string


# Setup Device & Logger
def setup_device():
   if torch.cuda.is_available():
       return TRAIN_DEVICE #torch.device("cuda")
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


# Logging Helpers
def log_classification_metrics(logger, split, y_true, y_pred, y_prob=None):
   report = classification_report(
       y_true,
       y_pred,
       target_names=["IP", "OP"],
       digits=4
   )


   logger.info(f"\n[{split}] Classification Report:\n{report}")


   if y_prob is not None:
       try:
           auc = roc_auc_score(y_true, y_prob[:, 1])
           logger.info(f"[{split}] ROC-AUC: {auc:.4f}")
       except Exception as e:
           logger.warning(f"[{split}] ROC-AUC computation failed: {e}")


# Prediction saving
def save_predictions(
   graph,
   y_true,
   y_pred,
   y_prob,
   split,
   trial_number,
   ip_f1,
   output_dir
):
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
           "ip_f1": ip_f1
       })


   pd.DataFrame(rows).to_csv(path, index=False)
   return path




# Pre-processing Text + Trigrams
nlp = spacy.load("en_core_web_sm")


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
   return [" ".join(toks[i:i+3]) for i in range(len(toks)-2)]


def trigram_vector(text, label, trigram_list, ip_set, op_set):
   trigs = extract_trigrams(text)
   vec = []
   for t in trigram_list:
       if label == "IP" and t in ip_set:
           vec.append(trigs.count(t))
       elif label == "OP" and t in op_set:
           vec.append(trigs.count(t))
       else:
           vec.append(0)
   return np.array(vec, dtype=np.float32)


def get_distinctive_trigram_vector(text, label, union_trigrams_list, ip_set, op_set):
   """
   For a given note, generate a feature vector (of fixed length equal to the union of significant trigrams)
   by counting occurrences of each trigram in the union.
   For a note with label 'IP', only counts for trigrams belonging to the IP set are kept (others set to 0),
   and vice versa for 'OP'.
   """
   trigrams_in_text = extract_trigrams(text)
   vector = []
   if label.upper() == 'IP':
       for trigram in union_trigrams_list:
           vector.append(trigrams_in_text.count(trigram) if trigram in ip_set else 0)
   elif label.upper() == 'OP':
       for trigram in union_trigrams_list:
           vector.append(trigrams_in_text.count(trigram) if trigram in op_set else 0)
   else:
       vector = [0] * len(union_trigrams_list)
   return np.array(vector, dtype=float)


# Analyze trigrams (lrt)
from scipy.stats import chi2


def likelihood_ratio_test(trigram, ip_trigrams_dict, op_trigrams_dict, total_IP, total_OP):
   """
   Calculate the log-likelihood ratio for a trigram
  
   Args:
   trigram (str): The trigram to analyze
   ip_trigrams_dict (dict): Trigram counts for ip group
   op_trigrams_dict (dict): Trigram counts for op group
   total_IP (int): Total count in ip group
   total_OP (int): Total count in op group
  
   Returns:
   float: Log-likelihood ratio
   """
   # Get trigram counts
   a = ip_trigrams_dict.get(trigram, 0)
   b = op_trigrams_dict.get(trigram, 0)
  
   n11 = a  # Trigram in IP group
   n12 = total_IP - a  # No trigram in IP group
   n21 = b  # Trigram in OP group
   n22 = total_OP - b  # No trigram in OP group
   n = n11 + n12 + n21 + n22  # Total observations
  
   # Compute log-likelihood under null hypothesis (words are independent)
   # Null hypothesis: P(trigram in IP) = P(trigram in OP)
   p11_null = (n11 + n21) / n
   p12_null = (n12 + n22) / n
  
   # Compute log-likelihood under alternative hypothesis
   p11_alt = n11 / (n11 + n12) if (n11 + n12) > 0 else 0
   p12_alt = n12 / (n11 + n12) if (n11 + n12) > 0 else 0
   p21_alt = n21 / (n21 + n22) if (n21 + n22) > 0 else 0
   p22_alt = n22 / (n21 + n22) if (n21 + n22) > 0 else 0
  
   def safe_log(x):
       return np.log(x) if x > 0 else 0
  
   # Log-likelihood under null hypothesis
   ll_null = (
       n11 * safe_log(p11_null) + n12 * safe_log(p12_null) +
       n21 * safe_log(p11_null) + n22 * safe_log(p12_null)
   )
  
   # Log-likelihood under alternative hypothesis
   ll_alt = (
       n11 * safe_log(p11_alt) + n12 * safe_log(p12_alt) +
       n21 * safe_log(p21_alt) + n22 * safe_log(p22_alt)
   )
  
   # Likelihood ratio test statistic (2 * log-likelihood difference)
   lr_statistic = 2 * (ll_alt - ll_null)
  
   return lr_statistic


def analyze_trigrams_from_csv(ip_csv_path, op_csv_path, alpha=0.01):
   """
   Analyze trigrams from CSV files using likelihood ratio test
  
   Args:
   ip_csv_path (str): Path to IP trigrams CSV file
   op_csv_path (str): Path to OP trigrams CSV file
   alpha (float): Significance level
  
   Returns:
   tuple: Sets of significant trigrams for each group
   """
   ip_df = pd.read_csv(ip_csv_path)
   op_df = pd.read_csv(op_csv_path)
  
   ip_trigrams_dict = dict(zip(ip_df['Trigram'], ip_df['IP_Count']))
   op_trigrams_dict = dict(zip(op_df['Trigram'], op_df['OP_Count']))
  
   total_IP = ip_df['IP_Count'].sum()
   total_OP = op_df['OP_Count'].sum()
  
   all_trigrams = set(ip_trigrams_dict.keys()).union(set(op_trigrams_dict.keys()))
  
   significant_trigrams = []
   ip_significant_set = set()
   op_significant_set = set()
  
   # Results dataframe to store detailed information
   results_data = []
  
   for trigram in all_trigrams:
       # Calculate likelihood ratio test statistic
       lr_statistic = likelihood_ratio_test(
           trigram, ip_trigrams_dict, op_trigrams_dict,
           total_IP, total_OP
       )
      
       dof = 1
      
       # Get p-value from chi-square distribution
       p_value = 1 - chi2.cdf(lr_statistic, dof)
      
       a = ip_trigrams_dict.get(trigram, 0)
       b = op_trigrams_dict.get(trigram, 0)
      
       # Check statistical significance
       is_significant = p_value < alpha
      
       # Store results
       results_data.append({
           'Trigram': trigram,
           'IP_Count': a,
           'OP_Count': b,
           'Likelihood_Ratio': lr_statistic,
           'P_Value': p_value,
           'Statistically_Significant': is_significant
       })
      
       # If statistically significant, categorize
       if is_significant:
           significant_trigrams.append(trigram)
          
           if a > b:
               ip_significant_set.add(trigram)
           elif b > a:
               op_significant_set.add(trigram)
           else:
               ip_significant_set.add(trigram)
               op_significant_set.add(trigram)
  
   results_df = pd.DataFrame(results_data)
  
   # Sort by p-value
   results_df = results_df.sort_values('P_Value')
  
   return significant_trigrams, ip_significant_set, op_significant_set, results_df




# GatorTron model class
class GatorTronEncoder(nn.Module):
   def __init__(self, model_id):
       super().__init__()
       self.model = AutoModel.from_pretrained(model_id)
       self.hidden_size = self.model.config.hidden_size


   def forward(self, input_ids, attention_mask):
       out = self.model(input_ids=input_ids, attention_mask=attention_mask)
       return out.last_hidden_state[:, 0, :]  # CLS


# GNN class
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
        return x
  


class GNNWrapper(nn.Module):
   def __init__(self, gnn, edge_index):
       super().__init__()
       self.gnn = gnn
       self.edge_index = edge_index


   def forward(self, x):
       data = Data(x=x, edge_index=self.edge_index)
       return self.gnn(data)




# Meta-Learning Trainer (Bilevel Optimization)
def train_meta_gnn(
    gnn,
    data,
    optimizer,
    criterion,
    meta_lr=1e-3,
    epochs=10,
    logger=None,
    alpha=1.5,   # TAM Hyperparameter
    beta=0.5     # TAM Hyperparameter
):
    device = data.x.device
    train_idx = (~data.meta_mask).nonzero(as_tuple=True)[0]
    meta_idx = data.meta_mask.nonzero(as_tuple=True)[0]

    weights = torch.nn.Parameter(
        torch.ones(len(train_idx), device=device) / len(train_idx)
    )
    meta_opt = torch.optim.SGD([weights], lr=meta_lr)

    for epoch in range(epochs):
        gnn.train()

        out = gnn(data)
        
        # Apply TAM adjustment to the raw GNN logits
        tam_out = compute_tam_logits(out, data.edge_index, data.y, alpha, beta)

        losses = torch.stack([
            criterion(tam_out[i].unsqueeze(0), data.y[i].unsqueeze(0))
            for i in train_idx
        ])

        weighted_loss = torch.sum(weights * losses)

        # Virtual step
        grads = torch.autograd.grad(
            weighted_loss,
            gnn.parameters(),
            create_graph=True
        )

        fast_weights = [
            p - optimizer.param_groups[0]['lr'] * g
            for p, g in zip(gnn.parameters(), grads)
        ]

        # Meta loss utilizing TAM
        meta_out = gnn(data)
        tam_meta_out = compute_tam_logits(meta_out, data.edge_index, data.y, alpha, beta)
        meta_loss = torch.mean(
            torch.stack([
                criterion(tam_meta_out[i].unsqueeze(0), data.y[i].unsqueeze(0))
                for i in meta_idx
            ])
        )

        meta_opt.zero_grad()
        meta_loss.backward()
        meta_opt.step()

        with torch.no_grad():
            new_weights = torch.clamp(weights, min=0)
            new_weights = new_weights / (new_weights.sum() + 1e-9)
            weights.copy_(new_weights)

        optimizer.zero_grad()

        out = gnn(data)
        tam_final_out = compute_tam_logits(out, data.edge_index, data.y, alpha, beta)

        final_losses = torch.stack([
            criterion(tam_final_out[i].unsqueeze(0), data.y[i].unsqueeze(0))
            for i in train_idx
        ])

        final_weighted_loss = torch.sum(weights.detach() * final_losses)

        final_weighted_loss.backward()
        optimizer.step()

        if logger and epoch % 2 == 0:
            logger.info(
                f"[Meta Epoch {epoch}] "
                f"Train Loss={weighted_loss.item():.4f} "
                f"Meta Loss={meta_loss.item():.4f}"
            )


def load_graph(json_path):
   with open(json_path) as f:
       return json.load(f)


def build_data_object(
   graph,
   gatortron_embs,
   empath,
   empath_cats,
   trigram_list,
   ip_set,
   op_set,
   reasoning_json,
   device
):
   reasoning_map = {}
   if reasoning_json:
       with open(reasoning_json) as f:
           for e in json.load(f):
               if "patient_id" in e and "embedding" in e:
                   reasoning_map[e["patient_id"]] = np.array(e["embedding"], dtype=np.float32)


   node_map = {n["patient_id"]: i for i, n in enumerate(graph["nodes"])}
   X, Y = [], []


   for i, n in enumerate(graph["nodes"]):
       text = n["collated_notes"]
       label = n["label"].strip().upper()
       y = 0 if label == "IP" else 1


       gt = gatortron_embs[i]


       # Empath
       emp = empath.analyze(text, normalize=True)
       emp_vec = np.array([emp.get(c, 0.0) for c in empath_cats])


       # Trigrams
       tri_vec = trigram_vector(text, label, trigram_list, ip_set, op_set)


       # Reasoning
       rsn = reasoning_map.get(n["patient_id"], np.zeros(384, dtype=np.float32))


       X.append(np.concatenate([gt, emp_vec, tri_vec, rsn]))
       Y.append(y)


   edges = [[node_map[e["source"]], node_map[e["target"]]] for e in graph["edges"]]
   edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()


   return Data(
       x=torch.tensor(np.stack(X), dtype=torch.float32).to(device),
       y=torch.tensor(Y, dtype=torch.long).to(device),
       edge_index=edge_index.to(device)
   )

from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F

def compute_tam_logits(logits, edge_index, labels, alpha=1.5, beta=0.5):
    """
    Adjusts raw logits using Topology-Aware Margin (TAM).
    Calculates ACM and ADM margins based on local neighborhood topology.
    """
    device = logits.device
    N, num_classes = logits.shape

    # Neighbor Label Distribution (D)
    y_onehot = F.one_hot(labels, num_classes=num_classes).float()
    edge_index_loop, _ = add_self_loops(edge_index, num_nodes=N)
    row, col = edge_index_loop

    neighbor_label_sums = torch.zeros(N, num_classes, device=device)
    neighbor_label_sums.scatter_add_(0, row.unsqueeze(-1).expand_as(y_onehot[col]), y_onehot[col])
    
    deg = degree(row, N, dtype=torch.float).view(-1, 1).clamp(min=1)
    D = neighbor_label_sums / deg

    # Class-wise Connectivity Matrix (C)
    C = torch.zeros((num_classes, num_classes), device=device)
    for k in range(num_classes):
        mask = (labels == k)
        if mask.sum() > 0:
            C[k] = D[mask].mean(dim=0)
        else:
            C[k] = 1.0 / num_classes

    # Compute Anomalous Connectivity-Aware Margin (ACM)
    ACM = torch.zeros_like(logits)
    for t in range(num_classes):
        c_yy = C[labels, labels]
        d_yy = D[torch.arange(N), labels] + 1e-9

        d_vt = D[:, t]
        c_yt = C[labels, t] + 1e-9

        ratio = (c_yy / d_yy) * (d_vt / c_yt)
        ratio = torch.clamp(ratio, min=1e-9)

        margin_t = -torch.clamp(torch.log(ratio), min=0.0)
        ACM[:, t] = margin_t

    ACM[torch.arange(N), labels] = 0.0

    # Compute Anomalous Distribution-Aware Margin (ADM)
    ADM = torch.zeros_like(logits)
    for t in range(num_classes):
        dist_vt = F.smooth_l1_loss(D, C[t].unsqueeze(0).expand(N, -1), reduction='none').mean(dim=1)
        dist_ty = F.smooth_l1_loss(C[t].unsqueeze(0).expand(N, -1), C[labels], reduction='none').mean(dim=1) + 1e-9

        adm_val = - (dist_vt / dist_ty)
        ADM[:, t] = adm_val

    ADM[torch.arange(N), labels] = 0.0

    return logits + (alpha * ACM) + (beta * ADM)

def train_gnn_tam(
    gnn,
    data,
    optimizer,
    criterion,
    epochs=50,
    logger=None,
    alpha=1.5,
    beta=0.5
):
    """Standard node classification training loop utilizing TAM loss."""
    for epoch in range(epochs):
        gnn.train()
        optimizer.zero_grad()

        # Forward pass to get raw logits
        out = gnn(data)

        # Apply Topology-Aware Margin adjustment
        tam_logits = compute_tam_logits(out, data.edge_index, data.y, alpha, beta)

        # Compute loss on all training nodes
        loss = criterion(tam_logits, data.y)
        loss.backward()
        optimizer.step()

        if logger and epoch % 10 == 0:
            logger.info(f"[Epoch {epoch}] Train Loss (TAM) = {loss.item():.4f}")

def objective_factory(
    train_data,
    test_data,
    gatortron,
    tokenizer,
    device,
    output_dir, logger, train_graph, test_graph, FEATURE_RANGES, empath_cats, trigram_list
):
    best_ip_f1 = {"val": 0.0}

    def objective(trial):
        hidden_dim = trial.suggest_int("hidden_dim", 32, 256)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        alpha = trial.suggest_float("alpha", 0.5, 3.0) # TAM Hyperparameter
        beta = trial.suggest_float("beta", 0.1, 1.0)   # TAM Hyperparameter

        gnn = MetaGNN(train_data.x.size(1), hidden_dim, 2).to(device)

        optimizer = optim.AdamW(
            list(gnn.parameters()) +
            [p for p in gatortron.parameters() if p.requires_grad],
            lr=lr
        )

        # Use CrossEntropyLoss since TAM requires unnormalized logits
        criterion = nn.CrossEntropyLoss()

        # Execute TAM training loop
        train_gnn_tam(
            gnn=gnn,
            data=train_data,
            optimizer=optimizer,
            criterion=criterion,
            epochs=50,
            logger=logger,
            alpha=alpha,
            beta=beta
        )

        gnn.eval()
        with torch.no_grad():
            # Train evaluation
            train_out = gnn(train_data)
            train_probs = F.softmax(train_out, dim=1)
            train_preds = train_probs.argmax(dim=1)

            log_classification_metrics(
                logger,
                split=f"Trial {trial.number} | TRAIN",
                y_true=train_data.y.cpu().numpy(),
                y_pred=train_preds.cpu().numpy(),
                y_prob=train_probs.cpu().numpy()
            )

            # Test evaluation
            test_out = gnn(test_data)
            test_probs = F.softmax(test_out, dim=1)
            test_preds = test_probs.argmax(dim=1)

            log_classification_metrics(
                logger,
                split=f"Trial {trial.number} | TEST",
                y_true=test_data.y.cpu().numpy(),
                y_pred=test_preds.cpu().numpy(),
                y_prob=test_probs.cpu().numpy()
            )

            report = classification_report(
                test_data.y.cpu().numpy(),
                test_preds.cpu().numpy(),
                output_dict=True,
                zero_division=0
            )

        ip_f1 = report["0"]["f1-score"]

        if ip_f1 > best_ip_f1["val"]:
            logger.info(
                f"New Best Model Found | Trial {trial.number} | "
                f"IP-F1: {ip_f1:.4f} (prev {best_ip_f1['val']:.4f})"
            )
            best_ip_f1["val"] = ip_f1

            torch.save(
                {
                    "gnn": gnn.state_dict(),
                    "gatortron": gatortron.state_dict(),
                    "trial": trial.number,
                    "params": trial.params,
                    "ip_f1": ip_f1
                },
                os.path.join(output_dir, "best_model_check.pt")
            )

            save_predictions(
                graph=train_graph, y_true=train_data.y.cpu().numpy(),
                y_pred=train_preds.cpu().numpy(), y_prob=train_probs.cpu().numpy(),
                split="train", trial_number=trial.number, ip_f1=ip_f1, output_dir=output_dir
            )

            save_predictions(
                graph=test_graph, y_true=test_data.y.cpu().numpy(),
                y_pred=test_preds.cpu().numpy(), y_prob=test_probs.cpu().numpy(),
                split="test", trial_number=trial.number, ip_f1=ip_f1, output_dir=output_dir
            )

        return ip_f1
    return objective


def main():
   device = setup_device()
   outdir = f"joint_run_ga_tam_logic_pure_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
   os.makedirs(outdir, exist_ok=True)
   logger = setup_logger(outdir)


   # Load graphs
   train_graph = load_graph("same_graph_train_bothmasked.json")
   test_graph = load_graph("same_graph_test_bothmasked.json")


   # GatorTron
   tokenizer = AutoTokenizer.from_pretrained("UFNLP/gatortron-base-2k")
   gatortron = GatorTronEncoder("UFNLP/gatortron-base-2k").to(device)


   for p in gatortron.model.parameters():
       p.requires_grad = False
   for layer in gatortron.model.encoder.layer[-4:]:
       for p in layer.parameters():
           p.requires_grad = True


   def encode(texts):
       embs = []
       gatortron.eval()
       with torch.no_grad():
           for t in texts:
               inp = tokenizer(
                   t,
                   truncation=True,
                   padding="max_length",
                   max_length=2000,
                   return_tensors="pt",
               ).to(device)


               embs.append(
                   gatortron(
                       input_ids=inp["input_ids"],
                       attention_mask=inp["attention_mask"],
                   )[0].squeeze(0).cpu().numpy()
               )
       return np.stack(embs)


   train_gt = encode([n["collated_notes"] for n in train_graph["nodes"]])
   test_gt = encode([n["collated_notes"] for n in test_graph["nodes"]])


   # Empath + Trigrams
   empath = Empath()
   empath_cats = sorted(empath.cats)


   # Load trigram data
   ip_csv_path = 'ip_specific_trigrams_masked_train_new_bothmasked.csv'
   op_csv_path = 'op_specific_trigrams_masked_train_new_bothmasked.csv'
  
   # Analyze trigrams
   print("Analyzing trigrams...")
   significant_trigrams, ip_set, op_set, results_df = analyze_trigrams_from_csv(
       ip_csv_path, op_csv_path
   )
   trigram_list = sorted(ip_set|op_set)
   ip_trigrams=ip_set
   op_trigrams=op_set
   # results_df.to_csv('resampling_trigram_likelihood_ratio_results_updated_train_gsage.csv', index=False)


#    ip_trigrams = set(pd.read_csv("ip_specific_trigrams_masked_train_new_bothmasked.csv")["Trigram"])
#    op_trigrams = set(pd.read_csv("op_specific_trigrams_masked_train_new_bothmasked.csv")["Trigram"])
#    trigram_list = sorted(ip_trigrams | op_trigrams)



   GT_DIM = train_gt.shape[1]
   EMPATH_DIM = len(empath_cats)
   TRIGRAM_DIM = len(trigram_list)

   train_data = build_data_object(train_graph, train_gt, empath, empath_cats,trigram_list, ip_trigrams, op_trigrams,"or_train_zsl_gptoss_embeddings.json", device)
   test_data = build_data_object(test_graph, test_gt, empath, empath_cats,trigram_list, ip_trigrams, op_trigrams,"or_test_zsl_gptoss_embeddings.json", device)


   REASON_DIM = train_data.x.size(1) - (GT_DIM + EMPATH_DIM + TRIGRAM_DIM)
   FEATURE_RANGES = {
       "gatortron": (0, GT_DIM),
       "empath": (GT_DIM, GT_DIM + EMPATH_DIM),
       "trigrams": (GT_DIM + EMPATH_DIM, GT_DIM + EMPATH_DIM + TRIGRAM_DIM),
       "reasoning": (GT_DIM + EMPATH_DIM + TRIGRAM_DIM, train_data.x.size(1))
   }
   logger.info(f"Feature ranges: {FEATURE_RANGES}")

   logger.info("========== Dataset Statistics ==========")
   logger.info(f"Train Nodes: {train_data.num_nodes}")
   logger.info(f"Train Edges: {train_data.num_edges}")
   logger.info(f"Train IP: {(train_data.y == 0).sum().item()}, OP: {(train_data.y == 1).sum().item()}")


   logger.info(f"Test Nodes: {test_data.num_nodes}")
   logger.info(f"Test Edges: {test_data.num_edges}")
   logger.info(f"Test IP: {(test_data.y == 0).sum().item()}, OP: {(test_data.y == 1).sum().item()}")
   logger.info("========================================")


   # Optuna study
   study = optuna.create_study(direction="maximize")
   study.optimize(
       objective_factory(
           train_data, test_data, gatortron,
           tokenizer, device, outdir, logger, train_graph, test_graph, FEATURE_RANGES, empath_cats, trigram_list
       ),
       n_trials=10
   )


   logger.info(f"Best trial: {study.best_trial.params}")


if __name__ == "__main__":
   main()
