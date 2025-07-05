# GNN-Based Drug-Target Binding Affinity Prediction

This project presents a web-based system to predict drug-target binding affinity using Graph Neural Networks (GNNs) and transformer-based protein sequence embeddings. It leverages PyTorch Geometric, RDKit, and Hugging Face models to analyze SMILES strings and protein FASTA sequences.

## 🚀 Features

- GNN model for drug molecular graphs
- Transformer-based protein encoder (e.g., ESM-2)
- Binding affinity prediction (e.g., pIC50, Kd, Ki)
- Web UI with:
  - Manual and batch input
  - 3D molecule visualization using 3Dmol.js
  - Protein-ligand contact maps
  - PDF download of results

## 📁 Project Structure

```

├── api.py                    # Flask backend for model inference
├── index.html               # Web UI frontend (Tailwind CSS + JS)
├── drug\_target\_gnn.pth      # Trained GNN model weights
├── drug\_target\_gnn.py       # Model architecture (DrugTargetGNN class)
├── utils.py                 # Helper functions for graph conversion & plots
├── requirements.txt         # Python dependencies
└── static/                  # JS libraries (3Dmol.js, Chart.js, etc.)

````

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/drug-affinity-gnn.git
cd drug-affinity-gnn
````

### 2. Install Dependencies

Make sure you have Python 3.8+ installed. Then run:

```bash
pip install -r requirements.txt
```

### 3. Run the Flask App

```bash
python api.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## 🔍 Example Input

* SMILES: `CC(=O)OC1=CC=CC=C1C(=O)O` (Aspirin)
* Protein (FASTA): `MENSDSAEDKTSGKLP...`

## 📈 Model Evaluation

| Dataset | MSE ↓ | CI ↑ | Pearson ↑ |
| ------- | ----- | ---- | --------- |
| DAVIS   | 0.28  | 0.87 | 0.82      |
| KIBA    | 0.33  | 0.85 | 0.80      |

## 📦 Datasets Used

* [DAVIS](https://www.bioinf.jku.at/research/DeepDTA/)
* [KIBA](https://www.bioinf.jku.at/research/DeepDTA/)

## 🧪 Tech Stack

* Python, Flask, PyTorch, PyTorch Geometric
* RDKit, Biopython, Hugging Face Transformers
* HTML, TailwindCSS, Chart.js, 3Dmol.js

## ✨ Future Scope

* Real-time screening via REST API
* Integration of 3D molecular dynamics
* Extended support for toxicity prediction

## 👨‍💻 Authors

* **Malreddy Vijay Reddy** 

