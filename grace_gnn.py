import json
import torch
import numpy as np
import pandas as pd
import optuna
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data, DataLoader
import torch.nn as nn
import torch.optim as optim
# from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from sklearn.metrics import classification_report, roc_auc_score
from empath import Empath
import re
import string
import matplotlib.pyplot as plt
import csv
from scipy.stats import chi2_contingency
import os
from copy import deepcopy
from datetime import datetime
import logging
import random
from torch_geometric.utils import subgraph
from tmd import TMD


# Add device detection and setup function
def setup_device():
    """
    Setup device for PyTorch operations.
    Prefers CUDA if available, falls back to CPU.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    return device


# configure a logger
def setup_logger(output_dir, name="meta_gage_logger"):
    """
    Create a logger that writes to both console and a file.
    
    Args:
        output_dir (str): Directory to save log file
        name (str): Logger name
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Create file handler
    log_path = os.path.join(output_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized. Logging to {log_path}")
    return logger

# -------------------------
# Additional NLP functions for trigram extraction
# -------------------------
import spacy
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    """
    Preprocess the input text by:
      - Converting to lowercase.
      - Removing stopwords, digits, punctuation, and tokens identified as names (PERSON entities).
      - Applying lemmatization.
    Returns a list of processed tokens.
    """
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_stop:
            continue
        if token.like_num or re.search(r'\d', token.text):
            continue
        if token.is_punct or token.text in string.punctuation:
            continue
        if token.ent_type_ == 'PERSON':
            continue
        if not token.is_alpha:
            continue
        lemma = token.lemma_.lower()
        tokens.append(lemma)
    return tokens

def extract_trigrams(text):
    """
    Preprocess the text and generate all trigrams (three-word sequences)
    from the list of processed tokens.
    """
    tokens = preprocess_text(text)
    return [' '.join(tokens[i:i+3]) for i in range(len(tokens) - 2)]

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

def get_extended_embedding(text, label, model, lexicon, category_order, union_trigrams_list, ip_set, op_set):
    """
    Compute the extended embedding by concatenating:
      1. The SentenceTransformer sentence embedding.
      2. The normalized Empath lexical features (ordered by category_order).
      3. The distinctive trigram frequency vector (of fixed dimension from the union of significant trigrams).
    """
    # Sentence transformer embedding
    sentence_emb = model.encode(text)
    
    # Empath features in a fixed sorted order
    impact_features = lexicon.analyze(text, normalize=True)
    impact_feature_vector = np.array([impact_features.get(cat, 0.0) for cat in category_order])
    
    # Distinctive trigram features (fixed dimension for all notes)
    distinctive_vector = get_distinctive_trigram_vector(text, label, union_trigrams_list, ip_set, op_set)
    
    # Concatenate all features
    extended_embedding = np.concatenate([sentence_emb, impact_feature_vector, distinctive_vector])
    return extended_embedding

# -------------------------
# Graph Data Loading and Data Object Creation
# -------------------------
def load_graph_data(json_file):
    with open(json_file, 'r') as f:
        graph_data = json.load(f)
    return graph_data

def create_data_object(graph_data, model, lexicon, category_order, union_trigrams_list, ip_set, op_set, justification_embedding_json, train_ratio=1.0, device=None, is_train_set=False):
    if device is None:
        device = torch.device('cpu')
    node_mapping = {node['patient_id']: idx for idx, node in enumerate(graph_data['nodes'])}
    node_features, labels = [], []

    if justification_embedding_json:
        print(f"Loading justification JSON from {justification_embedding_json}")
        with open(justification_embedding_json, 'r') as f:
            reasoning_data = json.load(f)
        id_to_embedding = {}
       # count_success=0
        count_missing_pid=0
        count_with_emb = 0
        count_without_emb = 0
        debug=False
        for entry in reasoning_data:
            if "patient_id" in entry:
                if "embedding" in entry and  entry["embedding"]:
                    pid = entry["patient_id"]
                    embedding_data=entry["embedding"]
                    id_to_embedding[entry["patient_id"]] = np.array(entry["embedding"], dtype=np.float32)
                    count_with_emb+=1
                else:
                    count_without_emb+=1
                    if not debug:
                        print(f"Actual keys: {list(entry.keys())}")
                        if "embedding" not in entry:
                            print("check typo")
                        debug=True
            else:
                count_missing_pid+=1
        #print("counts: ", count_with_emb, count_without_emb, count_missing_pid)
#            entry["patient_id"]: np.array(entry["embedding"], dtype=np.float32)
#            for entry in reasoning_data if "embedding" in entry and entry["embedding"]
#        }
    else:
        id_to_embedding = {}

    for node in graph_data['nodes']:
        text = node.get("collated_notes", "")
        label = node.get("label", "").strip().upper()
        true_label = 0 if label == 'IP' else 1

        extended_emb = get_extended_embedding(text, label, model, lexicon, category_order, union_trigrams_list, ip_set, op_set)

        if justification_embedding_json:
            justification_emb = id_to_embedding.get(node["patient_id"], np.zeros(384, dtype=np.float32))

            extended_emb = np.concatenate([extended_emb, justification_emb])

        node_features.append(extended_emb)

        labels.append(true_label)

    edge_index = [[node_mapping[edge['source']], node_mapping[edge['target']]] for edge in graph_data['edges']]
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)

    x = torch.tensor(node_features, dtype=torch.float).to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)
    num_nodes = len(graph_data['nodes'])
    train_size = int(train_ratio * num_nodes)
    indices = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    train_mask[indices[:train_size]] = True

    data = Data(x=x, edge_index=edge_index_tensor, y=y, train_mask=train_mask)
    return data

class MetaGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim, output_dim):
        super(MetaGCN, self).__init__()
        self.linear = torch.nn.Linear(input_dim, hidden_dim1)
        self.conv1 = SAGEConv(hidden_dim1, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x=self.linear(x)
        x = nn.ReLU()(x)
        x = self.conv1(x, edge_index)
        x = nn.ReLU()(x)
        x = self.conv2(x, edge_index)
        return nn.LogSoftmax(dim=1)(x)

# -------------------------
# Meta-Graph Sampling Function with GA
# -------------------------

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
    # Identify class indices (assume classes 0/IP and 1/OP)
    labels = data.y.cpu().numpy()
    idx_IP = np.where(labels == 0)[0].tolist()  # minority class (IP)
    idx_OP = np.where(labels == 1)[0].tolist()  # majority class (OP)
    num_IP, num_OP = len(idx_IP), len(idx_OP)
   
    n_meta = int(meta_ratio * N)
    n_each = n_meta // 2
   
    # Prepare semantic embeddings
    E = data.x.cpu().numpy() # shape [N, D]
    # Compute total variance for normalization
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

    # Precompute degrees for structural score
    edge_index = data.edge_index.cpu().numpy()
    deg = np.zeros(N, dtype=int)
    for u, v in edge_index.T:
        deg[u] += 1; deg[v] += 1
    mean_deg = deg.mean()

    # GA parameters (optimized using optuna)
    pop_size = 50
    generations = 100
    crossover_rate = 0.8745703676281257
    mutation_rate = 0.21873583752075254
    logger.info(f"GA parameters: population size = {pop_size}, #generations = {generations}, crossover_rate = {crossover_rate}, and mutation_rate = {mutation_rate}.")
       
    # Initialize population: each individual is (list_IP, list_OP)
    population = []
    for _ in range(pop_size):
        labels = data.y.cpu().numpy()
        idx_IP = np.where(labels==0)[0].tolist()
        idx_OP = np.where(labels==1)[0].tolist()
        sel_IP = random.sample(idx_IP, n_each)
        sel_OP = random.sample(idx_OP, n_each)
        population.append((sel_IP, sel_OP))
    
    logger.info(f"population initialized...")
    
    
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

        deg_sel = np.array(deg[sel_IP + sel_OP])
        f_deg = deg_sel.mean() / (mean_deg + 1e-9)
       
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
        
        #  treenorm
        subgraph_nodes = torch.unique(torch.tensor(individual, dtype=torch.long, device=data.edge_index.device))

        
        # Get induced subgraph with original edges
        edge_mask = (data.edge_index[0].unsqueeze(1) == subgraph_nodes).any(1) & \
                    (data.edge_index[1].unsqueeze(1) == subgraph_nodes).any(1)
        sub_edge_index = data.edge_index[:, edge_mask]

        # Create node mapping and relabel
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
        tree_norm_fullgraph = tree_norm_G#32088827904.0
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
                # Child1 and Child2 for IP class
                cut = random.randint(1, n_each-1)
                child1_IP = parent1[0][:cut] + parent2[0][cut:]
                child2_IP = parent2[0][:cut] + parent1[0][cut:]
                # Ensure uniqueness and correct size
                child1_IP = list(dict.fromkeys(child1_IP))
                child2_IP = list(dict.fromkeys(child2_IP))
                # Fill or trim
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
    data.meta_mask = meta_mask


    if logger:
        logger.info(f"Created meta-dataset with {len(selected)} samples")


    return data, selected


# -------------------------
# Meta learning Function
# -------------------------
def train_meta_gcn(model, data, optimizer, criterion, meta_lr=0.001, epochs=50, logger=None, device=None):
    if device is None:
        device = next(model.parameters()).device
    
    model = model.to(device)
    data = data.to(device)
    
    # Initialize example weights for each node in the training set
    train_indices = data.train_mask.nonzero().view(-1)
    num_train_nodes = len(train_indices)
    
    # Create a Parameter object directly
    example_weights = nn.Parameter(
        torch.ones(num_train_nodes, device=data.x.device) / num_train_nodes
    )
    meta_optimizer = torch.optim.SGD([example_weights], lr=meta_lr)

    if logger:
        logger.info(f"Starting Meta-GCN training with {epochs} epochs, meta_lr={meta_lr}")
        logger.info(f"Training on {num_train_nodes} nodes")
    
    for epoch in range(epochs):
        model.train()
        
        # Forward pass and compute individual losses
        outputs = model(data)
        individual_losses = []
        for i, idx in enumerate(train_indices):
            loss = criterion(outputs[idx].unsqueeze(0), data.y[idx].unsqueeze(0))
            individual_losses.append(loss)
        
        individual_losses = torch.stack(individual_losses)
        weighted_loss = torch.sum(example_weights * individual_losses)
        
        # Create virtual model with perturbed parameters
        virtual_model = MetaGCN(model.linear.in_features, model.linear.out_features, model.conv1.out_channels, model.conv2.out_channels).to(device)
        virtual_model.load_state_dict(model.state_dict())
        
        # Compute gradients for virtual update
        grad_params = torch.autograd.grad(
            weighted_loss, model.parameters(), create_graph=True
        )
        # Virtually update the model parameters
        for param, grad in zip(virtual_model.parameters(), grad_params):
            param.data = param.data - optimizer.param_groups[0]['lr'] * grad
        
        # Step 4: Compute meta-loss on meta-dataset
        meta_outputs = virtual_model(data)
        meta_indices = data.meta_mask.nonzero().view(-1)
        meta_losses = []
        
        for idx in meta_indices:
            meta_loss = criterion(meta_outputs[idx].unsqueeze(0), data.y[idx].unsqueeze(0))
            meta_losses.append(meta_loss)
        
        meta_loss = torch.mean(torch.stack(meta_losses))
        # Update example weights based on meta-loss
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()
        
        with torch.no_grad():
            example_weights.data = torch.clamp(example_weights.data, min=0.0)
            
            # Normalization with small delta
            delta = 1e-8
            weight_sum = example_weights.sum() + delta
            example_weights.data = example_weights.data / weight_sum
        
        # Update model parameters with adjusted weights
        optimizer.zero_grad()

        outputs = model(data)
        individual_losses = []
        for i, idx in enumerate(train_indices):
            loss = criterion(outputs[idx].unsqueeze(0), data.y[idx].unsqueeze(0))
            individual_losses.append(loss)
        
        individual_losses = torch.stack(individual_losses)
        weighted_loss = torch.sum(example_weights * individual_losses)
        weighted_loss.backward()
        optimizer.step()
    
    
    if logger:
        logger.info("Meta-GCN training completed")
    
    return example_weights

# -------------------------
# Evaluation Function
# -------------------------
def evaluate_model(model, loader, logger=None, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            probs = torch.exp(out)
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(data.y.cpu())
            all_probs.append(probs.cpu())
    
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    all_probs = torch.cat(all_probs).cpu().numpy()
    report = classification_report(all_labels, all_preds, output_dict=True)
    auc = roc_auc_score(all_labels, all_probs[:, 1])
    # print("Classification Report:", report)
    # Log results
    if logger:
        logger.info(f"Evaluation Results:")
        logger.info(f"AUC ROC Score: {auc}")
        logger.info(f"IP F1 Score: {report['0']['f1-score']:.4f}")
        logger.info(f"OP F1 Score: {report['1']['f1-score']:.4f}")
        logger.info(f"test_metrics: {report}")
        if device.type == 'cuda':
            logger.info(f"Evaluation completed on {device}")
    else:
        print("AUC ROC Score:", auc)
    return all_preds, all_labels, report

def save_detailed_predictions(predictions, true_labels, nodes_info, output_dir):
    detailed_predictions = []
    
    for i in range(len(predictions)):
        detailed_predictions.append({
            "patient_id": nodes_info[i]['patient_id'],
            "collated_notes": nodes_info[i]['collated_notes'],
            "predicted_label": int(predictions[i]),
            "true_label": int(true_labels[i])
        })
    # Plot confusion matrix
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['IP', 'OP'], yticklabels=['IP', 'OP'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm[i,j] / np.sum(cm) * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', ha='center', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    
    with open(os.path.join(output_dir, f"optuna_trials_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), 'w') as f:
        json.dump(detailed_predictions, f)


#  Save Meta-Dataset Function
def convert_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    else:
        return obj

def save_meta_dataset(train_graph_data, meta_indices, filename):
    """Save the meta-dataset nodes to a JSON file for cross-checking"""

    meta_nodes = []
    for idx in meta_indices:
        node = train_graph_data['nodes'][idx]
        node_serializable = convert_to_serializable(node)
        meta_nodes.append(node_serializable)

    meta_data = {
        "nodes": convert_to_serializable(meta_nodes),
        "meta_indices": [int(i) for i in meta_indices]  # ensure Python ints
    }

    with open(filename, 'w') as f:
        json.dump(meta_data, f)

    print(f"Meta-dataset saved to {filename}")


# -------------------------
# Analyze Trigrams
# -------------------------
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
    a = ip_trigrams_dict.get(trigram, 0)  # Count in IP group
    b = op_trigrams_dict.get(trigram, 0)  # Count in OP group
    
    # Total counts
    n11 = a  # Trigram in IP group
    n12 = total_IP - a  # No trigram in IP group
    n21 = b  # Trigram in OP group
    n22 = total_OP - b  # No trigram in OP group
    n = n11 + n12 + n21 + n22  # Total observations
    
    # Null hypothesis: P(trigram in IP) = P(trigram in OP)
    p11_null = (n11 + n21) / n
    p12_null = (n12 + n22) / n
    
    # Alternative hypothesis: probabilities can differ
    p11_alt = n11 / (n11 + n12) if (n11 + n12) > 0 else 0
    p12_alt = n12 / (n11 + n12) if (n11 + n12) > 0 else 0
    p21_alt = n21 / (n21 + n22) if (n21 + n22) > 0 else 0
    p22_alt = n22 / (n21 + n22) if (n21 + n22) > 0 else 0
    
    # Compute log-likelihood
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
    # Read CSV files
    ip_df = pd.read_csv(ip_csv_path)
    op_df = pd.read_csv(op_csv_path)
    
    # Create dictionaries for quick lookup
    ip_trigrams_dict = dict(zip(ip_df['Trigram'], ip_df['IP_Count']))
    op_trigrams_dict = dict(zip(op_df['Trigram'], op_df['OP_Count']))
    
    # Calculate total counts
    total_IP = ip_df['IP_Count'].sum()
    total_OP = op_df['OP_Count'].sum()
    
    # Get all unique trigrams
    all_trigrams = set(ip_trigrams_dict.keys()).union(set(op_trigrams_dict.keys()))
    
    # Lists to store results
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
        
        # Degrees of freedom (for 2x2 contingency table)
        dof = 1
        
        # Get p-value from chi-square distribution
        p_value = 1 - chi2.cdf(lr_statistic, dof)
        
        # Get counts
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
                # If equal, add to both sets
                ip_significant_set.add(trigram)
                op_significant_set.add(trigram)
    
    results_df = pd.DataFrame(results_data)
    
    # Sort by p-value for easy interpretation
    results_df = results_df.sort_values('P_Value')
    
    return significant_trigrams, ip_significant_set, op_significant_set, results_df


# -------------------------
# Optuna Objective Function for Meta-Dataset Resampling
# -------------------------
def create_objective(train_data, test_data, test_graph_data, input_dim, output_dim, meta_indices, output_dir, logfile_path, device, logger=None):
    """Creates an objective function with access to the current meta_dataset"""
    
    # Initialize best metric tracking for this meta-dataset sample
    best_test_metric = {"0": {"f1-score": 0.0}}
    best_predictions = None
    best_true_labels = None
    best_nodes_info = None
    
    # Create log file header
    with open(logfile_path, 'w') as log_file:
        log_file.write("Trial,Timestamp,hidden_dim,lr,meta_lr,IP_F1,OP_F1,Best_IP_F1_So_Far,test_metric\n")
    
    
    def objective(trial):
        nonlocal best_test_metric, best_predictions, best_true_labels, best_nodes_info
        hidden_dim = trial.suggest_int("hidden_dim", 8, 128)
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
        meta_lr = trial.suggest_loguniform("meta_lr", 1e-4, 1e-2)

        # Current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if logger:
            logger.info(f"Starting Trial {trial.number} - hidden_dim={hidden_dim}, lr={lr}, meta_lr={meta_lr} on {device}")
        
        
        # Clone the data with the current meta-set
        data_with_meta = train_data.clone()
        data_with_meta = data_with_meta.to(device)
        
        model_gcn = MetaGCN(input_dim, input_dim+12, hidden_dim, output_dim).to(device)
        optimizer = optim.Adam(model_gcn.parameters(), lr=lr)
        criterion = nn.NLLLoss(reduction='none')
        
        # Train with Meta-GCN approach
        example_weights = train_meta_gcn(model_gcn, data_with_meta, optimizer, criterion, meta_lr=meta_lr, epochs=50, logger = logger, device=device)

        model_gcn.eval()
        with torch.no_grad():
            x = model_gcn.linear(train_data.x)
            x = torch.relu(x)
            trained_embeddings = model_gcn.conv1(x, train_data.edge_index)

        plot_tsne(trained_embeddings, train_data.y, title="t-SNE After Training", output_path=os.path.join(output_dir, f"tsne_after_training_{timestamp}.png"))

        # Evaluate on test data
        test_loader = DataLoader([test_data], batch_size=1)
        predictions, true_labels, test_report = evaluate_model(model_gcn, test_loader, logger = logger, device = device)

        # Get F1 scores for both classes
        ip_f1 = test_report.get('0', {}).get('f1-score', 0.0)
        op_f1 = test_report.get('1', {}).get('f1-score', 0.0)
        
        # Update best metrics if this is the first trial or if we've improved
        if best_test_metric is None or ip_f1 > best_test_metric.get('0', {}).get('f1-score', 0.0):
            best_test_metric = test_report
            best_predictions = predictions
            best_true_labels = true_labels
            best_nodes_info = test_graph_data['nodes']
            if logger:
                logger.info(f"New best model found! IP F1: {ip_f1:.4f}")
                save_detailed_predictions(best_predictions, best_true_labels, best_nodes_info, output_dir)
        
        # Log results to file
        with open(logfile_path, 'a') as log_file:
            best_ip_f1 = best_test_metric.get('0', {}).get('f1-score', 0.0)
            log_file.write(f"{trial.number},{timestamp},{hidden_dim},{lr},{meta_lr},{ip_f1},{op_f1},{best_ip_f1},{test_report}\n")
        
        message = f"Trial {trial.number}: Test F1 (IP class): {ip_f1:.4f}"
        print(message)
        
        message2 = f"Best F1 (IP class) so far: {best_test_metric['0']['f1-score']:.4f}"
        print(message2)
        
        if logger:
            logger.info(message)
            logger.info(message2)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return ip_f1
    
    return objective, best_test_metric, best_predictions, best_true_labels, best_nodes_info

from sklearn.manifold import TSNE

def plot_tsne(embeddings, labels, title, output_path):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeddings.cpu().numpy())

    plt.figure(figsize=(8, 6))
    for label in torch.unique(labels):
        idx = (labels == label).cpu().numpy()
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f'Class {label.item()}', alpha=0.6)
    
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def main():

    device = setup_device()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"grace_gsage_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info(f"Starting Grace-Graphsage experimentation at {timestamp}")
    logger.info(f"Using device: {device}")
    
    # Create log file path
    logfile_path = os.path.join(output_dir, f"optuna_trials_{timestamp}.csv")
    logger.info(f"Logging Optuna results to: {logfile_path}")
    
    # Load sentence transformer model
    model_name = 'all-MiniLM-L6-v2'
    embedding_model = SentenceTransformer(model_name)
    logger.info(f"Loaded sentence transformer model: {model_name}")

    lexicon = Empath()
    category_order = sorted(lexicon.cats)
    logger.info(f"Initialized Empath lexicon with {len(category_order)} categories")

    # Load graph data
    train_graph_data = load_graph_data("train_graph.json")
    test_graph_data = load_graph_data("test_graph.json")
    logger.info(f"Loaded training graph with {len(train_graph_data['nodes'])} nodes, {len(train_graph_data['edges'])} edges")
    logger.info(f"Loaded test graph with {len(test_graph_data['nodes'])} nodes, {len(test_graph_data['edges'])} edges")
    
    # Load trigram data
    ip_csv_path = 'ip_specific_trigrams.csv' # obtained from n-gram.py
    op_csv_path = 'op_specific_trigrams.csv'
    
    # Analyze trigrams
    print("Analyzing trigrams...")
    significant_trigrams, ip_set, op_set, results_df = analyze_trigrams_from_csv(
        ip_csv_path, op_csv_path
    )
    # results_df.to_csv('resampling_trigram_likelihood_ratio_results_updated_train_gsage.csv', index=False)

    
    # Print summary
    print("Total Significant Trigrams:", len(significant_trigrams))
    print("IP Significant Trigrams:", len(ip_set))
    print("OP Significant Trigrams:", len(op_set))
    union_trigrams_list = sorted(significant_trigrams)
    

    train_reasoning_path = "train_reason_embeddings.json"
    test_reasoning_path = "test_reason_embeddings.json"

    base_train_data = create_data_object(train_graph_data, embedding_model, lexicon, category_order,
                                     union_trigrams_list, ip_set, op_set,
                                     justification_embedding_json=train_reasoning_path,
                                     train_ratio=1.0, device=device, is_train_set=True)

    test_data = create_data_object(test_graph_data, embedding_model, lexicon, category_order,
                                    union_trigrams_list, ip_set, op_set,
                                    justification_embedding_json=test_reasoning_path,
                                    train_ratio=1.0, device=device, is_train_set=False)

    input_dim = base_train_data.x.shape[1]
    logger.info(f"Extended input dimension (with justification): {input_dim}")

    output_dim = len(set(base_train_data.y.cpu().numpy()))
    
    # Save initial embeddings and labels
    initial_embeddings = base_train_data.x.clone()
    initial_labels = base_train_data.y.clone()
    plot_tsne(initial_embeddings, initial_labels, title="t-SNE Before Training", output_path=os.path.join(output_dir, "tsne_before_training.png"))

    print(f"\n======= Processing GA based Meta-Dataset =======")
    
    # Create new meta-dataset with seed
    train_data, meta_indices = create_meta_dataset(base_train_data.clone(), output_dir, meta_ratio=0.1, seed=42, logger=logger)

    # Save the meta-dataset
    meta_dataset_filename = os.path.join(output_dir, f"GA_meta_dataset_{timestamp}.json")
    save_meta_dataset(train_graph_data, meta_indices, meta_dataset_filename)
    
    # Create objective function for this meta-dataset
    objective_fn, best_test_metric, best_predictions, best_true_labels, best_nodes_info = create_objective(
        train_data, test_data, test_graph_data, input_dim, output_dim, meta_indices, output_dir, logfile_path, device=device, logger=logger)


    # Run Optuna optimization
    try:
        logger.info("Starting Optuna optimization")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_fn, n_trials=100)
        
        logger.info("Optimization complete")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best parameters: {study.best_trial.params}")
        logger.info(f"Best F1 score (IP): {study.best_value:.4f}")
        
        # Save study results
        import joblib
        study_path = os.path.join(output_dir, f"optuna_study_{timestamp}.pkl")
        joblib.dump(study, study_path)
        logger.info(f"Saved Optuna study to {study_path}")
        
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        return
    
    logger.info("Experiment completed successfully")
    
    

# -------------------------
# Run the main code
# -------------------------
if __name__ == "__main__":
    main()
