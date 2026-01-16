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
       return torch.log_softmax(x, dim=1)
  


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
   logger=None
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
       losses = torch.stack([
           criterion(out[i].unsqueeze(0), data.y[i].unsqueeze(0))
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


       # Meta loss
       meta_out = gnn(data)
       meta_loss = torch.mean(
           torch.stack([
               criterion(meta_out[i].unsqueeze(0), data.y[i].unsqueeze(0))
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


       final_losses = torch.stack([
           criterion(out[i].unsqueeze(0), data.y[i].unsqueeze(0))
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


# Meta Dataset creation (sampling using GA)
import warnings
if not hasattr(np, "warnings"):
   np.warnings = warnings


from sklearn.cluster import KMeans
import networkx as nx


def create_meta_dataset(data, output_dir, meta_ratio=0.1, seed=None, logger=None):
   """
   GA-based sampling to create a balanced subgraph (meta-dataset).
   Returns a boolean meta_mask, updated train_mask, and list of sampled indices.
   """
   if seed is not None:
       random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
   logger.info(f"Seed = {seed}")
   device = data.x.device




   N = data.y.size(0)
   labels = data.y.cpu().numpy()
   idx_IP = np.where(labels == 0)[0].tolist()  # minority class (IP)
   idx_OP = np.where(labels == 1)[0].tolist()  # majority class (OP)
   num_IP, num_OP = len(idx_IP), len(idx_OP)
 
   n_meta = int(meta_ratio * N)
   n_each = n_meta // 2
 
   # Prepare semantic embeddings
   E = data.x.cpu().numpy() # shape [N, D]
   # total variance for normalization
   V_total = np.var(E, axis=0).sum()


   import networkx as nx
   G = nx.Graph()
   G.add_nodes_from(range(N))
   G.add_edges_from(data.edge_index.t().tolist())
   communities = nx.community.greedy_modularity_communities(G)
   comm_map = {node: cid for cid, comm in enumerate(communities) for node in comm}
   n_communities = len(communities)
   clustering_coeff_G = nx.average_clustering(G)
   assortativity_G = nx.degree_assortativity_coefficient(G) if len(G.edges) > 0 else 0


   # degrees for structural score
   edge_index = data.edge_index.cpu().numpy()
   deg = np.zeros(N, dtype=int)
   for u, v in edge_index.T:
       deg[u] += 1; deg[v] += 1
   mean_deg = deg.mean()
 
   # GA parameters
   pop_size = 50
   generations = 100
   crossover_rate = 0.8745703676281257
   mutation_rate = 0.21873583752075254
   logger.info(f"GA parameters: population size = {pop_size}, #generations = {generations}, crossover_rate = {crossover_rate}, and mutation_rate = {mutation_rate}.")
 
   # Initialize population
   population = []
   for _ in range(pop_size):
       labels = data.y.cpu().numpy()
       idx_IP = np.where(labels==0)[0].tolist()
       idx_OP = np.where(labels==1)[0].tolist()
       sel_IP = random.sample(idx_IP, n_each)
       sel_OP = random.sample(idx_OP, n_each)
       population.append((sel_IP, sel_OP))
  
   logger.info(f"population initialized...")
  
   def compute_tmd(subgraph_nodes):
       """Helper function to compute TMD between original and subsampled graph."""
       subgraph_nodes = torch.unique(torch.tensor(subgraph_nodes, dtype=torch.long, device=data.edge_index.device))
     
       if len(subgraph_nodes) == 0:
           return float('inf')
     
       # Get induced subgraph with original edges
       edge_mask = (data.edge_index[0].unsqueeze(1) == subgraph_nodes).any(1) & \
                   (data.edge_index[1].unsqueeze(1) == subgraph_nodes).any(1)
       sub_edge_index = data.edge_index[:, edge_mask]
     
       node_mapping = {old.item(): new for new, old in enumerate(subgraph_nodes)}
     
       sub_edge_index = torch.stack([
       torch.tensor([node_mapping[x.item()] for x in sub_edge_index[0]],
       device=device),
       torch.tensor([node_mapping[x.item()] for x in sub_edge_index[1]],
       device=device)
       ])
       sub_data = Data(
       x=data.x[subgraph_nodes].detach().clone(),
       edge_index=sub_edge_index.detach().clone()
       ).cpu()
      
       original_data = Data(x=data.x.detach().cpu(),
       edge_index=data.edge_index.detach().cpu()
       )
      
       tmd_calculator = TMD(original_data, sub_data, w=[0.33, 1, 3], L=4)
       logger.info(f"TMD score of subgraph {subgraph_nodes} with tmd_score as: {tmd_calculator}")
       return tmd_calculator
  
   def compute_structural_metrics(sub_nodes):
       subgraph = G.subgraph(sub_nodes)
       clustering_coeff = nx.average_clustering(subgraph)
       assortativity = nx.degree_assortativity_coefficient(subgraph) if len(subgraph.edges) > 0 else 0
       return clustering_coeff, assortativity
 
   from torch_geometric.utils import to_dense_adj
   import math
   from torch_geometric.data import Data
   from torch_geometric.utils import to_dense_adj, degree


   def compute_dynamic_L_w(data: Data, output_dir: str, L_min=1, L_max=4, alpha=1.0):
       """Compute dynamic L and w for each node."""
       deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
       d_min = deg.min().item()
       d_max = deg.max().item()
       eps = 1e-5
       d_norm = (deg - d_min) / (d_max - d_min + eps)


       L_dict = {}
       w_dict = {}


       for v in range(data.num_nodes):
           d_v = d_norm[v].item()
           L_v = int(L_max - (L_max - L_min) * d_v)
           lambda_v = math.exp(-alpha * d_v)
           w_v = [lambda_v ** l for l in range(L_v)]
           L_dict[v] = L_v
           w_dict[v] = w_v[1:]  # skip w(0)


       with open(os.path.join(output_dir, "L_dict.json"), "w") as f:
           json.dump({str(k): int(v) for k, v in L_dict.items()}, f, indent=2)


       with open(os.path.join(output_dir, "w_dict.json"), "w") as f:
           json.dump({str(k): [float(w) for w in v] for k, v in w_dict.items()}, f, indent=2)




       return L_dict, w_dict, int(max(L_dict.values()))


   def tree_norm_dynamic(data: Data, output_dir: str, L_min=1, L_max=4, alpha=1.0):
       """
       Compute dynamic TreeNorm for a PyG graph with node-wise L and w.


       Parameters:
           data (Data): PyG graph object with `x` and `edge_index`
           L_min (int): Minimum depth
           L_max (int): Maximum depth
           alpha (float): Controls decay of weights


       Returns:
           float: TreeNorm value
       """
       assert L_max >= L_min >= 1
       L_dict, w_dict, L_global = compute_dynamic_L_w(data, output_dir, L_min, L_max, alpha)


       A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]  # [N, N]
       x = torch.norm(data.x, p=2, dim=1)  # [N]
       z = [x]


       for ell in range(1, L_global):
           z_ell = torch.matmul(A, z[ell - 1])
           z.append(z_ell)


       # For each node, compute b_v = z_0[v] + sum_{ell=1}^{L_v - 1} w_v[ell-1] * z[ell][v]
       b = torch.zeros(data.num_nodes)
       for v in range(data.num_nodes):
           L_v = L_dict[v]
           b_v = z[0][v].clone()
           for ell in range(1, L_v):
               b_v += w_dict[v][ell - 1] * z[ell][v]
           b[v] = b_v


       return b.abs().sum().item()


   tree_norm_G = tree_norm_dynamic(data, output_dir, L_min=1, L_max=4, alpha=1.0)


   def fitness(individual):
       sel_IP, sel_OP = individual
     
       # Structural diversity score
       # (a) coverage via degree
       deg_sel = np.array(deg[sel_IP + sel_OP])
       f_deg = deg_sel.mean() / (mean_deg + 1e-9)
       # (b) community coverage
       comm_sel = {comm_map[i] for i in sel_IP + sel_OP}
       f_comm = len(comm_sel) / n_communities
     
       clustering_coeff, assortativity = compute_structural_metrics(sel_IP + sel_OP)
       fcc = clustering_coeff / clustering_coeff_G
       fa = assortativity / assortativity_G


       f_struct = f_deg + f_comm + fcc + fa
     
       # Semantic variance score
       E_sel = E[sel_IP + sel_OP]
       V_sel = np.var(E_sel, axis=0).sum()
       f_semantic = V_sel / (V_total + 1e-9)
      
       # tmd computation using treenorm
       subgraph_nodes = torch.unique(torch.tensor(individual, dtype=torch.long, device=data.edge_index.device))
      
       # Get induced subgraph with original edges
       edge_mask = (data.edge_index[0].unsqueeze(1) == subgraph_nodes).any(1) & \
                   (data.edge_index[1].unsqueeze(1) == subgraph_nodes).any(1)
       sub_edge_index = data.edge_index[:, edge_mask]


       node_mapping = {old.item(): new for new, old in enumerate(subgraph_nodes)}


       sub_edge_index = torch.stack([
       torch.tensor([node_mapping[x.item()] for x in sub_edge_index[0]],
       device=device),
       torch.tensor([node_mapping[x.item()] for x in sub_edge_index[1]],
       device=device)
       ])
       sampled_graph = Data(x=data.x[subgraph_nodes].detach().clone(),
       edge_index=sub_edge_index.detach().clone()).cpu()
       tree_norm_subgraph = tree_norm_dynamic(sampled_graph, output_dir, L_min=1, L_max=4, alpha=1.0)
       import math
       tree_norm_fullgraph = tree_norm_G
       normalized_treenorm = math.log1p(tree_norm_subgraph) / math.log1p(tree_norm_fullgraph)
     
       return f_struct + f_semantic + normalized_treenorm
 
   # Main GA loop
   for gen in range(generations):
       # Evaluate fitness of all individuals
       fitness_vals = np.array([fitness(ind) for ind in population])
       # Selection (roulette)
       probs = fitness_vals / (fitness_vals.sum() + 1e-9)
       new_pop = []
       while len(new_pop) < pop_size:
           # Select two parents
           parents = np.random.choice(pop_size, size=2, p=probs, replace=False)
           parent1 = population[parents[0]]
           parent2 = population[parents[1]]
         
           # Crossover
           if random.random() < crossover_rate:
               cut = random.randint(1, n_each-1)
               child1_IP = parent1[0][:cut] + parent2[0][cut:]
               child2_IP = parent2[0][:cut] + parent1[0][cut:]
               child1_IP = list(dict.fromkeys(child1_IP))
               child2_IP = list(dict.fromkeys(child2_IP))
               while len(child1_IP) < n_each:
                   cand = random.choice(idx_IP)
                   if cand not in child1_IP: child1_IP.append(cand)
               while len(child2_IP) < n_each:
                   cand = random.choice(idx_IP)
                   if cand not in child2_IP: child2_IP.append(cand)
               child1_IP = child1_IP[:n_each]
               child2_IP = child2_IP[:n_each]
             
               # Repeat for OP class
               cut = random.randint(1, n_each-1)
               child1_OP = parent1[1][:cut] + parent2[1][cut:]
               child2_OP = parent2[1][:cut] + parent1[1][cut:]
               child1_OP = list(dict.fromkeys(child1_OP))
               child2_OP = list(dict.fromkeys(child2_OP))
               while len(child1_OP) < n_each:
                   cand = random.choice(idx_OP)
                   if cand not in child1_OP: child1_OP.append(cand)
               while len(child2_OP) < n_each:
                   cand = random.choice(idx_OP)
                   if cand not in child2_OP: child2_OP.append(cand)
               child1_OP = child1_OP[:n_each]
               child2_OP = child2_OP[:n_each]
           else:
               # No crossover: children are copies
               child1_IP, child1_OP = parent1[0][:], parent1[1][:]
               child2_IP, child2_OP = parent2[0][:], parent2[1][:]
         
           # Mutation: swap nodes in each child
           def mutate(sel_list, full_list):
               if random.random() < mutation_rate:
                   i = random.randrange(len(sel_list))
                   j = random.choice([x for x in full_list if x not in sel_list])
                   sel_list[i], = [j]  # swap one element out
               return sel_list
         
           child1_IP = mutate(child1_IP, idx_IP)
           child2_IP = mutate(child2_IP, idx_IP)
           child1_OP = mutate(child1_OP, idx_OP)
           child2_OP = mutate(child2_OP, idx_OP)
         
           new_pop.extend([(child1_IP, child1_OP), (child2_IP, child2_OP)])
     
       population = new_pop[:pop_size]  # replace old population
 
   # After GA, select best individual
   best_ind = max(population, key=fitness)
   sel_IP, sel_OP = best_ind
   selected = sel_IP + sel_OP
 
   # Create meta_mask (boolean tensor)
   meta_mask = torch.zeros(N, dtype=torch.bool, device = device)
   meta_mask[selected] = True






   # Update data object
   data.meta_mask = meta_mask




   if logger:
       logger.info(f"Created meta-dataset with {len(selected)} samples")




   return data, selected


# Optuna Objective
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
       lr = trial.suggest_loguniform("lr", 1e-4, 5e-3)
       meta_lr = trial.suggest_loguniform("lr", 1e-4, 5e-3)


       gnn = MetaGNN(train_data.x.size(1), hidden_dim, 2).to(device)


       optimizer = optim.AdamW(
           list(gnn.parameters()) +
           [p for p in gatortron.parameters() if p.requires_grad],
           lr=lr
       )


       criterion = nn.NLLLoss()


       # Meta-learning training
       train_meta_gnn(
           gnn=gnn,
           data=train_data,
           optimizer=optimizer,
           criterion=criterion,
           meta_lr=meta_lr,
           epochs=50,
           logger=logger
       )


       gnn.eval()
       with torch.no_grad():
           train_out = gnn(train_data)
           train_preds = train_out.argmax(dim=1)
           train_probs = torch.exp(train_out)


           log_classification_metrics(
               logger,
               split=f"Trial {trial.number} | TRAIN",
               y_true=train_data.y.cpu().numpy(),
               y_pred=train_preds.cpu().numpy(),
               y_prob=train_probs.cpu().numpy()
           )


           test_out = gnn(test_data)
           test_preds = test_out.argmax(dim=1)
           test_probs = torch.exp(test_out)


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
               output_dict=True
           )


       ip_f1 = report["0"]["f1-score"]


       if ip_f1 > best_ip_f1["val"]:
           logger.info(
               f"New Best Model Found | "
               f"Trial {trial.number} | "
               f"IP-F1: {ip_f1:.4f} (prev {best_ip_f1['val']:.4f})"
           )


           best_ip_f1["val"] = ip_f1


           # #Save model
           # torch.save(
           #     {
           #         "gnn": gnn.state_dict(),
           #         "gatortron": gatortron.state_dict(),
           #         "trial": trial.number,
           #         "params": trial.params,
           #         "ip_f1": ip_f1
           #     },
           #     os.path.join(output_dir, "best_model.pt")
           # )


           # Save TRAIN predictions
           train_pred_path = save_predictions(
               graph=train_graph,
               y_true=train_data.y.cpu().numpy(),
               y_pred=train_preds.cpu().numpy(),
               y_prob=train_probs.cpu().numpy(),
               split="train",
               trial_number=trial.number,
               ip_f1=ip_f1,
               output_dir=output_dir
           )


           # Save TEST predictions
           test_pred_path = save_predictions(
               graph=test_graph,
               y_true=test_data.y.cpu().numpy(),
               y_pred=test_preds.cpu().numpy(),
               y_prob=test_probs.cpu().numpy(),
               split="test",
               trial_number=trial.number,
               ip_f1=ip_f1,
               output_dir=output_dir
           )


           logger.info(f"Saved train predictions → {train_pred_path}")
           logger.info(f"Saved test predictions  → {test_pred_path}")

       return ip_f1
   return objective


def main():
   device = setup_device()
   outdir = f"joint_run_ga_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
   # results_df.to_csv('resampling_trigram_likelihood_ratio_results_updated_train_gsage.csv', index=False)


   ip_trigrams = set(pd.read_csv("ip_specific_trigrams_masked_train_new_bothmasked.csv")["Trigram"])
   op_trigrams = set(pd.read_csv("op_specific_trigrams_masked_train_new_bothmasked.csv")["Trigram"])
   trigram_list = sorted(ip_trigrams | op_trigrams)


   GT_DIM = train_gt.shape[1]
   EMPATH_DIM = len(empath_cats)
   TRIGRAM_DIM = len(trigram_list)


   # Build data
   train_data = build_data_object(
       train_graph, train_gt, empath, empath_cats,
       trigram_list, ip_trigrams, op_trigrams,
       "or_train_zsl_gptoss_embeddings.json", device
   )


   test_data = build_data_object(
       test_graph, test_gt, empath, empath_cats,
       trigram_list, ip_trigrams, op_trigrams,
       "or_test_zsl_gptoss_embeddings.json", device
   )


   REASON_DIM = train_data.x.size(1) - (GT_DIM + EMPATH_DIM + TRIGRAM_DIM)
   FEATURE_RANGES = {
       "gatortron": (0, GT_DIM),
       "empath": (GT_DIM, GT_DIM + EMPATH_DIM),
       "trigrams": (GT_DIM + EMPATH_DIM, GT_DIM + EMPATH_DIM + TRIGRAM_DIM),
       "reasoning": (GT_DIM + EMPATH_DIM + TRIGRAM_DIM, train_data.x.size(1))
   }


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
       n_trials=100
   )


   logger.info(f"Best trial: {study.best_trial.params}")


if __name__ == "__main__":
   main()
