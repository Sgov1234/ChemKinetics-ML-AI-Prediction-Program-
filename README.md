# BioCharge: Chemical Reaction & EDLC Analysis System
![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-blue.svg)

A high-throughput computational chemistry platform for the **BioCharge at CSUDH** research initiative. This system specializes in predicting reaction pathways, kinetics, and products for biomass pyrolysis, with a specific focus on developing **biochar-based EDLC supercapacitors**.

The platform includes a custom compound database for feedstocks and a novel 4-model consensus system for high-accuracy predictions.

## 🚀 What's New in V3.4

* **Realistic Kinetics:** Implemented an `ExponentialDecayKinetics` model. This simulates real-world catalyst deactivation and reactant consumption, making time-based predictions more accurate.
* **Robust Usability:** Added a `ManualSMILESHandler`. If a compound isn't found in any database, the system now prompts you to enter the SMILES string manually, preventing dead-ends.
* **Stable Batch Processing:** Integrated a `MemoryManager` that performs automatic garbage collection during large batch jobs to prevent memory-related crashes.
* **New Libraries:** Integrated **`chempy`** and **`chemlib`** for more advanced chemical analysis and stoichiometry.
* **Better Visuals:** Converted the metal selectivity plot from a heatmap to a clearer **bar chart** and changed kinetic graph x-axes to **minutes** for easier interpretation.

## 🔑 Key Features

* **4-Model Consensus Prediction:** Integrates four separate models (ReactionT5v2, Molecular Transformer, MolT5, and RXNMapper) to provide a single, high-confidence product prediction.
* **Specialized Feedstock Database:** Features a critical local database for Ailanthus alkaloids (e.g., *canthine-6-one*), lignin derivatives (e.g., *syringol*), and other pyrolysis products not found in standard PubChem libraries.
* **EDLC Capacitance Scoring:** A custom module (`CapacitancePotentialScorer`) that analyzes predicted products and reaction kinetics to score their potential for creating high-performance EDLCs.
* **High-Throughput Batch Processing:** Ingests and processes large-scale reaction data from Excel (`.xlsx`) files to run hundreds of simulations, saving results and visualizations automatically. Now includes memory management for stability.
* **Advanced Kinetic Modeling:** Calculates rate constants (k), conversion (%), and half-life (t½) using the Arrhenius equation **and** a more realistic exponential decay model.
* **Statistical Analysis:** Includes modules to perform statistical comparisons between different feedstocks (e.g., *Ailanthus* vs. *Brassica*) and assess Arrhenius fit quality.
* **Interactive CLI:** A menu-driven interface for single-reaction analysis, batch processing, 3D molecule viewing, and report generation.

## ⚙️ System Architecture

This is an integrated platform that runs from a single main script. The workflow is as follows:

1.  **Input:** The user provides reactants, conditions, and time via the interactive CLI or an Excel file.
2.  **Lookup:** The `EnhancedPubChemAPI` module retrieves SMILES, prioritizing the custom local database. If not found, the `ManualSMILESHandler` prompts the user for input.
3.  **Prediction:** The `FourModelConsensus` system predicts the most likely reaction products, weighted by model accuracy.
4.  **Kinetics:** The `ExponentialDecayKinetics` module calculates realistic kinetic parameters based on the specified condition (e.g., "pyrolysis" at 500°C) and time.
5.  **Analysis:** A suite of TIER 1-3 analysis modules score the reaction for confidence, metal selectivity, and EDLC potential.
6.  **Output:** The system generates:
    * A results Excel file (e.g., `reactions_results.xlsx`)
    * PNG plots for kinetics, performance, and resource usage
    * Interactive 3D molecule-viewer `.html` files

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone [your-repo-url]
    cd [your-repo-name]
    ```

2.  Ensure you have the `requirements.txt` file (which we corrected in our last message) that includes `chempy` and `chemlib`.

3.  Install all required dependencies with one command:
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Usage

Run the main application from your terminal:

```bash
python Biocharge_Chemistry_Analyzer.py
