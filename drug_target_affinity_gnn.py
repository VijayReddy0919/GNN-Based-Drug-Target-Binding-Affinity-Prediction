import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
from rdkit import Chem
import numpy as np
from sklearn.model_selection import train_test_split

# Validation functions for SMILES and protein sequences
def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def is_valid_protein(sequence):
    return all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence)

# Constants for feature dimensions
NUM_NODE_FEATURES = 34  # Matches drug_target_data.csv processing
NUM_AMINO_ACIDS = 20
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
MAX_PROTEIN_LENGTH = 1000  # Pad/truncate protein sequences to this length

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_features = []
    for atom in mol.GetAtoms():
        feature = np.zeros(NUM_NODE_FEATURES)
        # Atomic number (1-34)
        if atom.GetAtomicNum() <= 34:
            feature[atom.GetAtomicNum() - 1] = 1
        # Degree (0-4)
        degree = min(atom.GetDegree(), 4)
        feature[10 + degree] = 1
        # Formal charge (-5 to +5, mapped to 0-4)
        charge = atom.GetFormalCharge() + 5
        if 0 <= charge < 5:
            feature[15 + charge] = 1
        # Hybridization (SP, SP2, SP3, SP3D, SP3D2)
        hybrid = int(atom.GetHybridization())
        if 0 <= hybrid < 5:
            feature[20 + hybrid] = 1
        # Aromaticity
        feature[25] = atom.GetIsAromatic()
        # Explicit hydrogens (0-3)
        h_count = min(atom.GetTotalNumHs(), 3)
        feature[26 + h_count] = 1
        # Chirality
        chirality = 1 if atom.HasProp('_ChiralityPossible') else 0
        feature[30 + chirality] = 1
        atom_features.append(feature)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(np.array(atom_features), dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

def protein_to_tensor(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    # Truncate or pad sequence to MAX_PROTEIN_LENGTH
    sequence = sequence[:MAX_PROTEIN_LENGTH]
    tensor = torch.zeros(NUM_AMINO_ACIDS, MAX_PROTEIN_LENGTH)
    for i, aa in enumerate(sequence):
        if aa in aa_to_idx:
            tensor[aa_to_idx[aa], i] = 1
    return tensor

class DrugTargetDataset(Dataset):
    def __init__(self, csv_file):
        try:
            data = pd.read_csv(csv_file)
        except Exception as e:
            raise ValueError(f"Failed to load CSV file: {str(e)}")
        # Validate SMILES and protein sequences
        valid_idx = [
            i for i in range(len(data))
            if is_valid_smiles(data['smiles'][i]) and is_valid_protein(data['protein_sequence'][i])
        ]
        self.smiles = [data['smiles'][i] for i in valid_idx]
        self.proteins = [data['protein_sequence'][i] for i in valid_idx]
        self.affinities = [data['affinity'][i] for i in valid_idx]
        if not valid_idx:
            raise ValueError("No valid drug-target pairs found in the dataset")
        print(f"Loaded {len(self.smiles)} valid drug-target pairs from {csv_file}")

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        protein = self.proteins[idx]
        affinity = self.affinities[idx]
        graph = smiles_to_graph(smiles)
        protein_tensor = protein_to_tensor(protein)
        return graph, protein_tensor, affinity

def custom_collate_fn(batch):
    graphs, proteins, affinities = zip(*batch)
    # Filter out None graphs
    valid_indices = [i for i, g in enumerate(graphs) if g is not None]
    if not valid_indices:
        return None, None, None
    graphs = [graphs[i] for i in valid_indices]
    proteins = [proteins[i] for i in valid_indices]
    affinities = [affinities[i] for i in valid_indices]
    # Batch graphs using Batch.from_data_list
    graph_batch = Batch.from_data_list(graphs)
    # Stack protein tensors (now fixed-size due to padding)
    protein_batch = torch.stack(proteins, dim=0)
    affinity_batch = torch.tensor(affinities, dtype=torch.float)
    return graph_batch, protein_batch, affinity_batch

class DrugTargetGNN(nn.Module):
    def __init__(self):
        super(DrugTargetGNN, self).__init__()
        self.conv1 = nn.Linear(NUM_NODE_FEATURES, HIDDEN_DIM)
        self.conv2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.protein_embed = nn.Conv1d(NUM_AMINO_ACIDS, EMBEDDING_DIM, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(HIDDEN_DIM + EMBEDDING_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, drug_data, protein_data):
        x = F.relu(self.conv1(drug_data.x))
        x = F.relu(self.conv2(x))
        drug_embed = scatter_mean(x, drug_data.batch, dim=0)
        protein_embed = F.relu(self.protein_embed(protein_data))
        protein_embed = protein_embed.mean(dim=2)
        combined = torch.cat([drug_embed, protein_embed], dim=1)
        x = F.relu(self.fc1(combined))
        x = self.fc2(x)
        return x

def train_model():
    dataset = DrugTargetDataset('drug_target_data.csv')
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    # Use torch_geometric.loader.DataLoader with custom collate_fn
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    model = DrugTargetGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    loss_log = []
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            drug_graph, protein_tensor, affinity = batch
            if drug_graph is None:  # Skip invalid batches
                continue
            drug_graph = drug_graph.to(device)
            protein_tensor = protein_tensor.to(device)
            affinity = affinity.to(device).float()
            optimizer.zero_grad()
            out = model(drug_graph, protein_tensor).squeeze()
            loss = loss_fn(out, affinity)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)  # Avoid division by zero

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                drug_graph, protein_tensor, affinity = batch
                if drug_graph is None:
                    continue
                drug_graph = drug_graph.to(device)
                protein_tensor = protein_tensor.to(device)
                affinity = affinity.to(device).float()
                out = model(drug_graph, protein_tensor).squeeze()
                loss = loss_fn(out, affinity)
                test_loss += loss.item()
        test_loss /= max(len(test_loader), 1)

        loss_log.append([epoch + 1, train_loss, test_loss])
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    torch.save(model.state_dict(), 'drug_target_gnn.pth')
    pd.DataFrame(loss_log, columns=['Epoch', 'Train_Loss', 'Test_Loss']).to_csv('loss_log.csv', index=False)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model()