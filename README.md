Chemical Reaction Pathway Prediction
A comprehensive collection of pre-trained machine learning and AI models for chemical reaction pathway kinetic and product prediction, molecule structure visualization and modeling.
Overview
This repository contains implementations of various state-of-the-art ML/AI models specifically designed for computational chemistry applications. The project focuses on predicting chemical reaction pathways, kinetics, and products using different deep learning approaches.
Features

Multiple Model Architectures: Implementations of ChemProp, DeepChem, T5Chem, and GT4SD models
Reaction Pathway Prediction: Predict possible reaction pathways and intermediates
Kinetic Analysis: Estimate reaction kinetics and rate constants
Product Prediction: Forecast reaction products and yields
Molecular Visualization: Structure visualization and modeling capabilities
Pre-trained Models: Leverage existing trained models for immediate use

Models Included
ChemProp Prediction Model

File: ChemProp Prediction Model.py
Graph neural network approach for molecular property prediction
Optimized for chemical reaction analysis

DeepChem Model

File: DeepChem Model.py
Deep learning framework for drug discovery and chemical informatics
Comprehensive molecular modeling capabilities

T5Chem Prediction Model

File: T5Chem Prediction Model.py
Transformer-based model for chemical synthesis prediction
Natural language processing approach to chemistry

GT4SD Models

Files:

GT4SD T5Chem Prediction model V.2.py
GT4SD T5Chem Prediction model V.3.py


Generative Toolkit for Scientific Discovery models
Advanced chemical space exploration and generation

FutureHouse Phoenix

File: FutureHouse Phoenix Prediction Code.py
Cutting-edge prediction algorithms for chemical reactions
Enhanced accuracy for complex reaction mechanisms

Installation
Prerequisites
bash# Python 3.7+ required
pip install torch torchvision
pip install rdkit-pypi
pip install deepchem
pip install chemprop
pip install gt4sd
Dependencies
The project requires the following main libraries:

PyTorch
RDKit
DeepChem
ChemProp
GT4SD
NumPy
Pandas
Matplotlib (for visualization)

Usage
Basic Example
python# Example usage for chemical reaction prediction
python ChemProp\ Prediction\ Model.py --input "your_molecule_smiles"
Model-Specific Usage
Each model can be run independently:
bash# ChemProp model
python "ChemProp Prediction Model.py"

# DeepChem model
python "DeepChem Model.py"

# T5Chem model
python "T5Chem Prediction Model.py"

# GT4SD models
python "GT4SD T5Chem Prediction model V.3.py"
System Architecture
Reaction Prediction Pipeline

Compound Lookup:

Primary: PubChem API for SMILES retrieval
Fallback: USPTO MIT dataset (cached locally)
Input validation with RDKit


AI Model Inference:

Model-specific tokenization and encoding
Sequence-to-sequence generation for products
SMILES validation and filtering


Kinetic Modeling:

Arrhenius equation: k = A × exp(-Ea/RT)
Integrated rate laws (0th, 1st, 2nd order)
Numerical integration for complex kinetics


Visualization:

Time-series concentration plots
2D molecular structure rendering
Rate law and equation annotations



Output Files Generated

reaction_progress.png - Kinetic simulation plot
reactants_visualization.png - 2D reactant structures
products_visualization.png - 2D product structures
uspto_data_cache.csv - Cached molecular database (auto-generated)

Reaction Conditions
ConditionPre-exponential Factor (A)Activation Energy (Ea)Reaction OrderPyrolysis1×10¹⁵ s⁻¹180 kJ/mol1Combustion1×10¹⁴ L/(mol·s)120 kJ/mol2Electrochemical1×10¹⁰ s⁻¹40 kJ/mol1Ideal1×10¹² s⁻¹80 kJ/mol1
Applications

Drug Discovery: Predict drug-target interactions and ADMET properties
Synthetic Chemistry: Plan synthetic routes and predict yields
Materials Science: Design new materials with desired properties
Chemical Safety: Assess reaction hazards and environmental impact
Process Optimization: Optimize reaction conditions and catalysts

Bug fixes
New model implementations
Performance improvements
Documentation updates

Requirements

Python 3.7+
CUDA-compatible GPU (recommended)
8GB+ RAM
Scientific computing libraries (NumPy, SciPy, Pandas)

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

ChemProp developers for the graph neural network implementation
DeepChem team for the comprehensive chemical modeling framework
T5Chem creators for transformer-based chemistry models
GT4SD team for generative scientific discovery tools
FutureHouse for advanced prediction algorithms
