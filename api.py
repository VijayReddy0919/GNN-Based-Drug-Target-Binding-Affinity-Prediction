# Switch matplotlib to a thread-safe backend before any imports
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive rendering
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import torch_scatter
import logging
import traceback
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Constants
NUM_NODE_FEATURES = 34
NUM_AMINO_ACIDS = 20
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
MAX_PROTEIN_LENGTH = 1000
MODEL_PATH = 'drug_target_gnn.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Validation functions
def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception as e:
        logger.debug(f"Invalid SMILES: {smiles}, error: {str(e)}")
        return False

def is_valid_protein(sequence):
    try:
        return all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence)
    except Exception as e:
        logger.debug(f"Invalid protein sequence: {sequence[:50]}..., error: {str(e)}")
        return False

def smiles_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug(f"Failed to create molecule from SMILES: {smiles}")
            return None
        atom_features = []
        for atom in mol.GetAtoms():
            feature = np.zeros(NUM_NODE_FEATURES)
            if atom.GetAtomicNum() <= 34:
                feature[atom.GetAtomicNum() - 1] = 1
            degree = min(atom.GetDegree(), 4)
            feature[10 + degree] = 1
            charge = atom.GetFormalCharge() + 5
            if 0 <= charge < 5:
                feature[15 + charge] = 1
            hybrid = int(atom.GetHybridization())
            if 0 <= hybrid < 5:
                feature[20 + hybrid] = 1
            feature[25] = atom.GetIsAromatic()
            h_count = min(atom.GetTotalNumHs(), 3)
            feature[26 + h_count] = 1
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
    except Exception as e:
        logger.error(f"Error in smiles_to_graph for {smiles}: {str(e)}")
        return None

def protein_to_tensor(sequence):
    try:
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
        sequence = sequence[:MAX_PROTEIN_LENGTH]
        tensor = torch.zeros(NUM_AMINO_ACIDS, MAX_PROTEIN_LENGTH)
        for i, aa in enumerate(sequence):
            if aa in aa_to_idx:
                tensor[aa_to_idx[aa], i] = 1
        return tensor
    except Exception as e:
        logger.error(f"Error in protein_to_tensor for sequence {sequence[:50]}...: {str(e)}")
        return None

class DrugTargetGNN(nn.Module):
    def __init__(self):
        super(DrugTargetGNN, self).__init__()
        self.conv1 = nn.Linear(NUM_NODE_FEATURES, HIDDEN_DIM)
        self.conv2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.protein_embed = nn.Conv1d(NUM_AMINO_ACIDS, EMBEDDING_DIM, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(HIDDEN_DIM + EMBEDDING_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, drug_data, protein_data):
        try:
            x = F.relu(self.conv1(drug_data.x))
            x = F.relu(self.conv2(x))
            drug_embed = torch_scatter.scatter_mean(x, drug_data.batch, dim=0)
            protein_embed = F.relu(self.protein_embed(protein_data))
            protein_embed = protein_embed.mean(dim=2)
            combined = torch.cat([drug_embed, protein_embed], dim=1)
            x = F.relu(self.fc1(combined))
            x = self.fc2(x)
            return x
        except Exception as e:
            logger.error(f"Error in model forward: {str(e)}")
            raise

# Load model
if not os.path.exists(MODEL_PATH):
    logger.error(f"Model file {MODEL_PATH} not found")
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found")
try:
    model = DrugTargetGNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}\n{traceback.format_exc()}")
    raise

def normalize_affinity(value, metric):
    try:
        if metric in ['IC50', 'Ki', 'Kd', 'EC50']:
            return -np.log10(float(value) * 1e-9 + 1e-10)
        return float(value)
    except Exception as e:
        logger.error(f"Error normalizing affinity {value} for metric {metric}: {str(e)}")
        return None

def generate_molecule_image(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        logger.debug(f"Molecule image generated, length: {len(img_str)}")
        return img_str
    except Exception as e:
        logger.error(f"Error generating molecule image for {smiles}: {str(e)}")
        return None

def generate_affinity_plot(affinities):
    try:
        if not affinities or len(affinities) < 1:
            logger.warning("No affinities to plot")
            return None
        plt.figure(figsize=(8, 6))
        plt.hist(affinities, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Predicted Affinity (pIC50)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Affinities')
        buffered = BytesIO()
        plt.savefig(buffered, format='png', bbox_inches='tight')
        plt.close()
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        logger.debug(f"Affinity plot generated, length: {len(img_str)}")
        return img_str
    except Exception as e:
        logger.error(f"Error generating affinity plot: {str(e)}")
        return None

def generate_contact_map(smiles, protein):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        protein_coords = []
        for i in range(min(len(protein), 100)):
            protein_coords.append([i * 1.0, 0.0, 0.0])
        protein_coords = np.array(protein_coords)

        ligand_coords = []
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            ligand_coords.append([pos.x, pos.y, pos.z])
        ligand_coords = np.array(ligand_coords)

        DISTANCE_THRESHOLD = 5.0
        contact_map = np.zeros((len(protein_coords), len(ligand_coords)))
        for i, p_coord in enumerate(protein_coords):
            for j, l_coord in enumerate(ligand_coords):
                dist = np.linalg.norm(p_coord - l_coord)
                if dist < DISTANCE_THRESHOLD:
                    contact_map[i, j] = 1

        plt.figure(figsize=(8, 6))
        plt.imshow(contact_map, cmap='Blues', interpolation='nearest')
        plt.xlabel('Ligand Atom Index')
        plt.ylabel('Protein Residue Index')
        plt.title('Protein-Ligand Contact Map')
        plt.colorbar(label='Contact (1 = <5Å)')
        buffered = BytesIO()
        plt.savefig(buffered, format='png', bbox_inches='tight')
        plt.close()
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        logger.debug(f"Contact map generated, length: {len(img_str)}")
        return img_str
    except Exception as e:
        logger.error(f"Error generating contact map: {str(e)}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    logger.debug("Received predict request")
    try:
        data = request.get_json()
        if not data or not isinstance(data, list) or len(data) == 0:
            logger.error("Invalid or no data provided in predict request")
            return jsonify({'error': 'Invalid or no data provided; expected a list with at least one item'}), 400

        results = []
        all_affinities = []

        for item in data:
            smiles = item.get('smiles')
            protein = item.get('protein')
            metric = item.get('metric', 'pIC50')

            logger.debug(f"Processing SMILES: {smiles[:50]}..., Protein: {protein[:50]}..., Metric: {metric}")

            if not smiles or not protein:
                logger.warning("Missing SMILES or protein in item")
                results.append({'error': 'Missing SMILES or protein sequence'})
                continue

            smiles_list = [smiles] if isinstance(smiles, str) else smiles
            if not is_valid_protein(protein):
                logger.warning(f"Invalid protein sequence: {protein[:50]}...")
                results.append({'error': 'Invalid protein sequence'})
                continue
            if not all(is_valid_smiles(sm) for sm in smiles_list):
                logger.warning(f"Invalid SMILES: {smiles_list}")
                results.append({'error': 'One or more invalid SMILES strings'})
                continue

            drug_graphs = [smiles_to_graph(sm) for sm in smiles_list]
            drug_graphs = [g for g in drug_graphs if g is not None]
            if not drug_graphs:
                logger.warning("Failed to convert any SMILES to graph")
                results.append({'error': 'Failed to convert any SMILES to graph'})
                continue

            protein_tensor = protein_to_tensor(protein)
            if protein_tensor is None:
                logger.warning(f"Failed to convert protein sequence: {protein[:50]}...")
                results.append({'error': 'Failed to convert protein sequence'})
                continue

            try:
                drug_batch = Batch.from_data_list(drug_graphs).to(device)
                protein_tensor = protein_tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    prediction = model(drug_batch, protein_tensor)
                affinities = prediction.squeeze().cpu().numpy()
                affinity = float(np.mean(affinities))
                logger.debug(f"Raw prediction: {affinity}")
            except Exception as e:
                logger.error(f"Error during model prediction: {str(e)}\n{traceback.format_exc()}")
                results.append({'error': f'Model prediction failed: {str(e)}'})
                continue

            normalized_affinity = normalize_affinity(affinity, metric)
            if normalized_affinity is None:
                logger.warning(f"Invalid affinity value for normalization: {affinity}")
                results.append({'error': 'Invalid affinity value for normalization'})
                continue

            logger.debug(f"Normalized affinity: {normalized_affinity}")
            outcome = 'High' if normalized_affinity > 7 else 'Low' if normalized_affinity < 5 else 'Moderate'
            molecule_image = generate_molecule_image(smiles_list[0]) if smiles_list else None
            contact_map = generate_contact_map(smiles_list[0], protein) if smiles_list else None
            all_affinities.append(normalized_affinity)

            results.append({
                'smiles': smiles_list,
                'protein': protein,
                'affinity': normalized_affinity,
                'outcome': outcome,
                'molecule_image': molecule_image,
                'contact_map': contact_map
            })

        # Generate a single affinity plot for all valid predictions
        plot_image = generate_affinity_plot(all_affinities) if all_affinities else None

        logger.debug(f"Returning {len(results)} results")
        return jsonify({'results': results, 'plot_image': plot_image})

    except Exception as e:
        logger.error(f"Unexpected error in /predict: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/generate_sdf', methods=['POST'])
def generate_sdf():
    try:
        data = request.get_json()
        smiles = data.get('smiles')
        if not smiles or not is_valid_smiles(smiles):
            return jsonify({'error': 'Invalid SMILES string'}), 400

        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        sdf_str = Chem.MolToMolBlock(mol)
        return jsonify({'sdf': sdf_str})
    except Exception as e:
        logger.error(f"Error generating SDF: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Failed to generate SDF: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'API is running', 'model_loaded': os.path.exists(MODEL_PATH)}), 200

if __name__ == '__main__':
    logger.info("Starting Flask server on http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)