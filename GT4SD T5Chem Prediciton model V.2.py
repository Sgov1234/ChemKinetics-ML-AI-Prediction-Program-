import requests
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import time
import pandas as pd
import io
import os
from scipy.integrate import solve_ivp

# --- Dependencies for Hugging Face Transformers ---
# You will need to install these libraries:
# pip install transformers
# pip install torch # or tensorflow, depending on your setup
# pip install sentencepiece # <--- Add this line
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import torch # Assuming PyTorch backend
# ------------------------------------------------


class ReactionPredictor:
    def __init__(self):
        # --- Initialize the AI Model (Hugging Face T5) ---
        self.hf_tokenizer = None
        self.hf_model = None
        self.load_hf_t5_model() # Call the method to load the Hugging Face T5 model
        # -------------------------------------------------

        # Constants for reaction rate calculation
        self.R = 8.314  # Gas constant (J/mol·K)

        # Dictionary mapping reaction conditions to typical parameters
        self.condition_params = {
            "pyrolysis": {"A": 1e15, "Ea": 180000, "order": 1},
            "combustion": {"A": 1e14, "Ea": 120000, "order": 2},
            "electrochemical": {"A": 1e10, "Ea": 40000, "order": 1},
            "ideal": {"A": 1e12, "Ea": 80000, "order": 1}
        }

        # Cache for USPTO data to avoid repeated downloads
        self.uspto_data = None

        # Simulation time in seconds - will be set by user input
        self.simulation_time = None

    def load_hf_t5_model(self):
        """Loads the Hugging Face T5 model and tokenizer."""
        print("Loading Hugging Face T5 model: GT4SD/multitask-text-and-chemistry-t5-base-augm...")
        try:
            # Load tokenizer using T5Tokenizer explicitly
            self.hf_tokenizer = T5Tokenizer.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-augm")
            # Load model directly from the specified model name
            self.hf_model = AutoModelForSeq2SeqLM.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-augm")
            self.hf_model.eval() # Set model to evaluation mode

            # Move model to GPU if available
            if torch.cuda.is_available():
                self.hf_model.to('cuda')
                print("Model moved to GPU.")

            print("Hugging Face T5 model and tokenizer loaded successfully.")

        except Exception as e:
            print(f"Error loading Hugging Face T5 model: {e}")
            print("Hugging Face T5 prediction will not be available.")
            self.hf_tokenizer = None
            self.hf_model = None


    # --- Removed previous AI setup methods ---
    # def setup_gemini_api(self): pass
    # def load_t5chem_model(self): pass
    # def load_deepchem_model(self): pass
    # def setup_phoenix_api(self): pass
    # ---------------------------------------


    def get_smiles_from_pubchem(self, compound_name):
        """Retrieve SMILES notation for a compound from PubChem"""
        print(f"Searching PubChem for {compound_name}...")
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/property/CanonicalSMILES/JSON"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                smiles = data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
                print(f"Found in PubChem: {smiles}")
                return smiles
            else:
                print(f"Not found in PubChem. Error code: {response.status_code}")
                return self.get_smiles_from_uspto(compound_name)
        except Exception as e:
            print(f"Error retrieving from PubChem: {e}")
            return self.get_smiles_from_uspto(compound_name)

    def load_uspto_data(self):
        """Load USPTO MIT dataset for chemical information"""
        if self.uspto_data is not None:
            return

        print("Loading USPTO MIT dataset (this may take a moment)...")
        try:
            # Try to load from a local cached file first
            if os.path.exists('uspto_data_cache.csv'):
                self.uspto_data = pd.read_csv('uspto_data_cache.csv')
                print("Loaded USPTO data from local cache.")
                return

            # If not available locally, download a sample from the USPTO dataset
            url = "https://dataverse.harvard.edu/api/access/datafile/3407241"  # USPTO SMILES dataset
            response = requests.get(url)

            if response.status_code == 200:
                # Process and save the data
                self.uspto_data = pd.read_csv(io.StringIO(response.text), sep='\t')
                # Save a local cache to avoid repeated downloads
                self.uspto_data.to_csv('uspto_data_cache.csv', index=False)
                print("USPTO data downloaded and cached locally.")
            else:
                print(f"Failed to download USPTO data. Error code: {response.status_code}")
                self.uspto_data = pd.DataFrame(columns=['NAME', 'SMILES'])
        except Exception as e:
            print(f"Error loading USPTO data: {e}")
            self.uspto_data = pd.DataFrame(columns=['NAME', 'SMILES'])


    def get_smiles_from_uspto(self, compound_name):
        """Retrieve SMILES notation from USPTO MIT dataset"""
        print(f"Searching USPTO dataset for {compound_name}...")
        try:
            self.load_uspto_data()

            # Search for the compound in the dataset
            compound_lower = compound_name.lower()
            for index, row in self.uspto_data.iterrows():
                if 'NAME' in row and isinstance(row['NAME'], str):
                    if compound_lower in row['NAME'].lower():
                        print(f"Found in USPTO: {row['SMILES']}")
                        return row['SMILES']

            print(f"Compound not found in USPTO dataset.")
            return None

        except Exception as e:
            print(f"Error searching USPTO data: {e}")
            return None


    def predict_reaction_hf_t5(self, reactants_smiles):
        """Predict reaction products using the Hugging Face T5 model."""
        if self.hf_model is None or self.hf_tokenizer is None:
            print("Hugging Face T5 model or tokenizer not loaded. Cannot perform prediction.")
            # Fallback to returning the first reactant as product (or empty list)
            return [reactants_smiles[0]] if reactants_smiles else []

        print("Querying Hugging Face T5 model...")

        try:
            # The GT4SD model expects input in a specific format, often "reactant_smiles >> product_smiles"
            # For prediction, we provide the reactant side and let the model generate the product side.
            reactants_string = ".".join(reactants_smiles)
            input_text = f"{reactants_string} >>" # Input format for prediction

            # Tokenize the input
            # Ensure return_tensors is set correctly ('pt' for PyTorch)
            inputs = self.hf_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

            # Move inputs to the same device as the model
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            # Generate prediction
            # You might need to tune generation parameters like num_beams, max_length, etc.
            with torch.no_grad(): # Use torch.no_grad() for inference
                output_ids = self.hf_model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=512, # Adjust max length as needed
                    num_beams=5, # Using beam search for potentially better results
                    early_stopping=True
                )

            # Decode the generated output IDs
            predicted_smiles_string = self.hf_tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # The model output should be the product SMILES.
            # Split the SMILES string by dot to get individual products.
            products_smiles = [s.strip() for s in predicted_smiles_string.split('.') if s.strip()]

            print(f"Hugging Face T5 predicted raw output: {predicted_smiles_string}")
            print(f"Parsed product SMILES: {products_smiles}")

            # Validate and filter predicted products
            valid_products = []
            for smiles in products_smiles:
                 try:
                     mol = Chem.MolFromSmiles(smiles)
                     if mol:
                         valid_products.append(smiles)
                     else:
                         print(f"Invalid SMILES predicted by Hugging Face T5: '{smiles}'. Skipping.")
                 except:
                     print(f"Error parsing SMILES predicted by Hugging Face T5: '{smiles}'. Skipping.")
                     continue

            if not valid_products:
                 print("No valid product SMILES found from Hugging Face T5 prediction. Generating default product.")
                 # Fallback to returning the first reactant as product
                 valid_products = [reactants_smiles[0]] if reactants_smiles else []

            return valid_products

        except Exception as e:
            print(f"Error during Hugging Face T5 reaction prediction: {e}")
            # Fallback to returning the first reactant as product
            return [reactants_smiles[0]] if reactants_smiles else []


    # --- Keep the rest of the methods as they are ---
    # calculate_rate_constant, use_custom_k, calculate_reaction_rate,
    # print_stoichiometric_equation, print_rate_law, integrated_rate_law,
    # simulate_reaction, plot_reaction_progress, visualize_molecules,
    # run_prediction_and_simulation (modified below), main (modified below)

    def calculate_rate_constant(self, temperature, condition):
        """Calculate reaction rate constant using Arrhenius equation"""
        params = self.condition_params.get(condition, self.condition_params["ideal"])
        A = params["A"]  # Pre-exponential factor
        Ea = params["Ea"]  # Activation energy (J/mol)

        # Arrhenius equation: k = A * exp(-Ea/RT)
        k = A * np.exp(-Ea / (self.R * temperature))
        return k, params["order"]

    def use_custom_k(self, k_value, condition):
        """Use a user-provided rate constant instead of calculating it"""
        order = self.condition_params.get(condition, self.condition_params["ideal"])["order"]
        return float(k_value), order

    def calculate_reaction_rate(self, k, concentrations, order):
        """Calculate reaction rate based on rate constant and concentrations"""
        if order == 0:
            return k
        elif order == 1:
            return k * concentrations[0]
        elif order == 2:
            if len(concentrations) > 1:
                return k * concentrations[0] * concentrations[1]
            else:
                return k * concentrations[0]**2
        else:
            # Fallback for orders > 2, assuming it depends on the first reactant only
            # This part might need refinement based on specific higher-order reactions
            return k * concentrations[0]**order


    def print_stoichiometric_equation(self, reactant_names, product_names, initial_reactant_concs_nominal=None, total_product_conc_final=None):
        """Print the *predicted chemical transformation* and optionally nominal reactant concentrations."""
        reactant_terms = []
        if initial_reactant_concs_nominal:
            for i, name in enumerate(reactant_names):
                # Ensure reactant_names[i] exists if initial_reactant_concs_nominal is shorter (though typically same length)
                if i < len(reactant_names):
                    reactant_terms.append(f"{initial_reactant_concs_nominal[i]:.2f} M {reactant_names[i]}")
                else: # Should not happen if lists are managed correctly
                    reactant_terms.append(f"{initial_reactant_concs_nominal[i]:.2f} M Unknown_Reactant")
        else:
            reactant_terms = list(reactant_names)

        # product_names already include their formulas from a previous step
        product_terms = list(product_names)

        equation = " + ".join(reactant_terms) + " → " + " + ".join(product_terms)
        print("\nPredicted Chemical Transformation:") # Changed title
        print(equation)
        if initial_reactant_concs_nominal:
            print("(Nominal initial concentrations shown for reactants. This equation reflects the T5 model's predicted outcome and may not be atom-balanced with simple 1:1 reactant coefficients.)")
        if total_product_conc_final is not None:
             print(f"(Total concentration of product(s) formed at simulation end, based on primary reactant conversion: {total_product_conc_final:.2f} M)")
        return equation


    def print_rate_law(self, reactant_names, order, k):
        """Print the rate law equation"""
        if order == 0:
            rate_law = f"Rate = k = {k:.2e} mol/(L·s)"
        elif order == 1:
            rate_law = f"Rate = k[{reactant_names[0]}] = {k:.2e}[{reactant_names[0]}] s⁻¹"
        elif order == 2:
            if len(reactant_names) > 1: # Assumes elementary reaction A + B -> P
                rate_law = f"Rate = k[{reactant_names[0]}][{reactant_names[1]}] = {k:.2e}[{reactant_names[0]}][{reactant_names[1]}] L/(mol·s)"
            else: # Assumes elementary reaction 2A -> P or A -> P with second order dependence on A
                rate_law = f"Rate = k[{reactant_names[0]}]² = {k:.2e}[{reactant_names[0]}]² L/(mol·s)"
        else:
            # Generic rate law for other orders, focusing on the first reactant
            rate_law = f"Rate = k[{reactant_names[0]}]^{order} = {k:.2e}[{reactant_names[0]}]^{order}"
            # Units would vary here, L^(order-1) / (mol^(order-1)·s)
            # For simplicity, not adding complex unit calculation here, assuming s^-1 for pseudo-order if not 1 or 2.

        print("\nRate Law:")
        print(rate_law)
        return rate_law


    def integrated_rate_law(self, order, initial_conc, k, time_points):
        """Calculate concentration using integrated rate laws"""
        concentrations = np.zeros_like(time_points, dtype=float)

        if order == 0:
            # Zero-order: [A] = [A]₀ - kt
            concentrations = initial_conc - k * time_points
            # Ensure no negative concentrations
            concentrations = np.maximum(concentrations, 0)

        elif order == 1:
            # First-order: [A] = [A]₀ * exp(-kt)
            concentrations = initial_conc * np.exp(-k * time_points)

        elif order == 2:
            # Second-order: 1/[A] = 1/[A]₀ + kt (assuming reaction 2A -> P or A+B -> P where [A]0=[B]0 and we track A)
            # If it's A+B and [A]0 != [B]0, the integrated law is more complex and not handled here.
            # This simulation simplifies to tracking one reactant's concentration or pseudo-order.
            concentrations = initial_conc / (1 + initial_conc * k * time_points)

        else:
            # For other orders, use numerical integration (pseudo-order for the first reactant)
            def reaction_rate(t, C):
                # Ensure C is not negative before exponentiation if order is not an integer
                C_safe = np.maximum(C, 0)
                return -k * (C_safe ** order)

            solution = solve_ivp(
                reaction_rate,
                [time_points[0], time_points[-1]],
                [initial_conc],
                t_eval=time_points,
                method='RK45', # Using a common explicit solver
                dense_output=True
            )

            concentrations = solution.sol(time_points)[0]
            concentrations = np.maximum(concentrations, 0) # Ensure no negative concentrations from numerical solution

        return concentrations


    def simulate_reaction(self, initial_conc, k, time_points, order):
        """Simulate reaction progress over time using integrated rate law"""
        # Assumes the kinetic model tracks the disappearance of the primary reactant (or a pseudo-order reactant)
        reactant_conc = self.integrated_rate_law(order, initial_conc, k, time_points)

        # Calculate total product concentration based on stoichiometry (assuming 1:1 conversion from the tracked reactant)
        # This means for every mole of the tracked reactant consumed, one "mole equivalent" of products is formed.
        product_conc = initial_conc - reactant_conc

        return reactant_conc, product_conc


    def plot_reaction_progress(self, time_points, reactant_conc, product_conc, reactant_names, products, temperature, condition, k, order):
        """Plot the reaction progress over time"""
        plt.figure(figsize=(12, 8))

        # Plot reactant concentration decrease
        # Assuming reactant_conc refers to the first reactant if multiple are named for kinetics.
        plt.plot(time_points, reactant_conc, 'b-', linewidth=2, label=f"{reactant_names[0]} (Reactant)")

        # Plot total product formation
        plt.plot(time_points, product_conc, 'r-', linewidth=2, label="Total Products")

        # Calculate half-life if applicable (meaningful for 1st order or pseudo-1st order)
        if order == 1:
            half_life = np.log(2) / k
            if half_life <= time_points[-1] and half_life > 0 : # Ensure half-life is within plot range and positive
                plt.axvline(x=half_life, color='g', linestyle='--', label=f"Half-life (t½) = {half_life:.2f}s")

        # Format time axis with appropriate units
        if time_points[-1] > 3600 * 2: # Show hours if more than 2 hours
            time_display = time_points / 3600
            plt.xlabel('Time (hours)', fontsize=12)
            tick_locs = np.linspace(0, time_points[-1], 11) # 11 ticks for hours
            tick_labels = [f"{t/3600:.1f}" for t in tick_locs]
        elif time_points[-1] > 120: # Show minutes if more than 120 seconds
            time_display = time_points / 60
            plt.xlabel('Time (minutes)', fontsize=12)
            tick_locs = np.linspace(0, time_points[-1], 11) # 11 ticks for minutes
            tick_labels = [f"{t/60:.1f}" for t in tick_locs]
        else:
            time_display = time_points
            plt.xlabel('Time (s)', fontsize=12)
            tick_locs = np.linspace(0, time_points[-1], 11) # 11 ticks for seconds
            tick_labels = [f"{t:.1f}" for t in tick_locs]

        plt.xticks(tick_locs, tick_labels)


        plt.ylabel('Concentration (mol/L)', fontsize=12)
        reaction_time_text = f"{time_points[-1]/3600:.1f} hours" if time_points[-1] > 3600 * 2 else \
                             f"{time_points[-1]/60:.1f} minutes" if time_points[-1] > 120 else \
                             f"{time_points[-1]:.0f} seconds"
        plt.title(f'Reaction Progress: {" + ".join(reactant_names)} at {temperature}K ({condition}) over {reaction_time_text}\n' +
                 f'Rate constant k = {k:.2e}, Reaction order = {order} (for {reactant_names[0]})', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)

        # Add rate law equation to the plot
        # This refers to the kinetic model used for simulation
        if order == 0:
            rate_law_text = f"Rate = k = {k:.2e}"
            integrated_law_text = f"[{reactant_names[0]}] = [{reactant_names[0]}]₀ - kt"
        elif order == 1:
            rate_law_text = f"Rate = k[{reactant_names[0]}] (k = {k:.2e})"
            integrated_law_text = f"[{reactant_names[0]}] = [{reactant_names[0]}]₀e^(-kt)"
        elif order == 2: # Assuming pseudo-second order or 2A -> P
            rate_law_text = f"Rate = k[{reactant_names[0]}]² (k = {k:.2e})"
            integrated_law_text = f"1/[{reactant_names[0]}] = 1/[{reactant_names[0]}]₀ + kt"
        else:
            rate_law_text = f"Rate = k[{reactant_names[0]}]^{order} (k = {k:.2e})"
            integrated_law_text = f"Numerical integration for [{reactant_names[0]}]"

        plt.figtext(0.5, 0.01, f"Kinetic Model: {rate_law_text}\nIntegrated Form: {integrated_law_text}",
                   ha="center", fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})

        # Save the plot to a temporary file and return the filename
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to make space for figtext
        filename = "reaction_progress.png"
        plt.savefig(filename)
        plt.show()
        print(f"Saved reaction progress plot to {filename}")
        return filename


    def visualize_molecules(self, reactants_smiles, products_smiles):
        """Create visual representations of reactants and products"""
        reactant_mols = [Chem.MolFromSmiles(smiles) for smiles in reactants_smiles if smiles]
        product_mols = [Chem.MolFromSmiles(smiles) for smiles in products_smiles if smiles]

        # Filter out None values (failed SMILES conversions)
        reactant_mols = [mol for mol in reactant_mols if mol is not None]
        product_mols = [mol for mol in product_mols if mol is not None]

        if not reactant_mols and not product_mols:
            print("Warning: No valid molecules to visualize.")
            return None, None # Return two Nones for reactant and product filenames

        # Generate 2D coordinates for better visualization
        for mol_list in [reactant_mols, product_mols]:
            for mol in mol_list:
                if mol: # Check if molecule is valid before generating coordinates
                    AllChem.Compute2DCoords(mol)

        reactant_filename = None
        product_filename = None

        try:
            if reactant_mols:
                reactant_img = Draw.MolsToGridImage(reactant_mols, molsPerRow=min(4, len(reactant_mols)), subImgSize=(250, 250),
                                                   legends=[f"Reactant {i+1}: {reactants_smiles[i]}" for i in range(len(reactant_mols))])
                if reactant_img:
                    reactant_filename = "reactants_visualization.png"
                    reactant_img.save(reactant_filename)
                    print(f"Saved reactant visualization to {reactant_filename}")
            else:
                print("No valid reactants to visualize.")

            if product_mols:
                product_img = Draw.MolsToGridImage(product_mols, molsPerRow=min(4, len(product_mols)), subImgSize=(250, 250),
                                                 legends=[f"Product {i+1}: {products_smiles[i]}" for i in range(len(product_mols))])
                if product_img:
                    product_filename = "products_visualization.png"
                    product_img.save(product_filename)
                    print(f"Saved product visualization to {product_filename}")
            else:
                print("No valid products to visualize.")

            return reactant_filename, product_filename

        except Exception as e:
            print(f"Failed to create molecule visualization using MolsToGridImage: {e}")
            return None, None


    def run_prediction_and_simulation(self, reactant_names, condition, temperature, simulation_time, custom_k=None, user_product_input=None):
        """
        Main method to run the entire prediction and simulation workflow.
        Optionally uses user-provided products instead of AI prediction.
        """
        print(f"\n*** Starting reaction analysis for {' + '.join(reactant_names)} under {condition} conditions at {temperature}K ***\n")

        # Set simulation time for this run
        self.simulation_time = simulation_time

        # Convert common names to SMILES for reactants
        reactants_smiles = []
        valid_reactant_names = []
        for name in reactant_names:
            smiles = self.get_smiles_from_pubchem(name)
            if smiles:
                reactants_smiles.append(smiles)
                valid_reactant_names.append(name) # Keep track of names for which SMILES were found
                print(f"Retrieved SMILES for {name}: {smiles}")
            else:
                print(f"Failed to retrieve SMILES for {name}. This reactant will be excluded from prediction and kinetics.")

        if not reactants_smiles: # Check if any valid SMILES were found
            print("Error: Need at least one valid reactant with SMILES notation to proceed.")
            return

        # Use valid_reactant_names for subsequent operations that require names matching the SMILES
        reactant_names_for_kinetics = valid_reactant_names

        # --- Determine products: Use user input if provided, otherwise predict with Hugging Face T5 ---
        products_smiles = []
        product_source = "Predicted by Hugging Face T5" # Default source

        if user_product_input and user_product_input.strip():
            print("\nUsing user-provided products...")
            user_product_names_input = [name.strip() for name in user_product_input.split(",")]
            temp_products_smiles = []
            for name in user_product_names_input:
                smiles = self.get_smiles_from_pubchem(name) # Try to get SMILES if name is provided
                if smiles:
                    temp_products_smiles.append(smiles)
                    print(f"Retrieved SMILES for user-provided product {name}: {smiles}")
                else: # If PubChem fails, assume the input might be a SMILES string itself
                    try:
                        mol = Chem.MolFromSmiles(name) # Check if 'name' is a valid SMILES
                        if mol:
                            temp_products_smiles.append(name)
                            print(f"Using user-provided SMILES: {name}")
                        else:
                            print(f"Could not retrieve SMILES for user-provided product '{name}' and it's not a valid SMILES. Skipping.")
                    except: # RDKit error
                         print(f"Error parsing user-provided product '{name}' as SMILES. Skipping.")


            if not temp_products_smiles:
                 print("Error: No valid SMILES found or retrieved for user-provided products. Cannot proceed with kinetics.")
                 return # Exit if user provided products but none were valid
            products_smiles = temp_products_smiles
            product_source = "Provided by User"

        else:
            print("\nPredicting reaction products using Hugging Face T5 model...")
            # Call the Hugging Face T5 prediction method
            products_smiles = self.predict_reaction_hf_t5(reactants_smiles)

            if not products_smiles: # Prediction might return first reactant on failure
                print("Hugging Face T5 prediction failed or returned no distinct products. Using first reactant as product.")
                # predict_reaction_hf_t5 already handles fallback to [reactants_smiles[0]]
                if not reactants_smiles: # Should be caught earlier, but as a safeguard
                    print("Critical error: No reactants available for fallback.")
                    return


        print(f"Products SMILES ({product_source}): {products_smiles}")

        # Get product names (using molecular formula as placeholder)
        product_display_names = []
        for i, smiles in enumerate(products_smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
                    product_display_names.append(f"Product_{i+1} ({formula})")
                else:
                    # This case handles invalid SMILES that RDKit cannot parse
                    product_display_names.append(f"Product_{i+1} (Invalid SMILES: '{smiles[:20]}...')") # Clarified
            except Exception as e: # Catching generic Exception is broad, but okay for this context
                print(f"Warning: Could not derive formula for product SMILES '{smiles}': {e}")
                product_display_names.append(f"Product_{i+1} (Formula Error for SMILES: '{smiles[:20]}...')") # Clarified


        # --- Proceed with kinetics calculations and visualization using the determined products ---
        # The kinetic model will primarily track the first reactant if multiple are provided.
        # The 'order' from condition_params applies to this primary reactant or a pseudo-order context.

        # Calculate reaction rate constant
        if custom_k is not None:
            k, reaction_order = self.use_custom_k(custom_k, condition)
            print(f"\nUsing custom rate constant k = {k:.2e}")
        else:
            k, reaction_order = self.calculate_rate_constant(temperature, condition)
            print(f"\nCalculated rate constant:")
            print(f"- Rate constant k = {k:.2e}")

        print(f"- Reaction order = {reaction_order} (applied to '{reactant_names_for_kinetics[0]}' or as pseudo-order)")


        # Print rate law equation based on the first reactant for simplicity if multiple reactants
        self.print_rate_law(reactant_names_for_kinetics, reaction_order, k)


        # Determine the applicable integrated rate law
        if reaction_order == 1:
            integrated_law_text_desc = f"[{reactant_names_for_kinetics[0]}] = [{reactant_names_for_kinetics[0]}]₀e^(-kt)"
            if k > 0: # Half-life is undefined if k=0
                half_life = np.log(2) / k
                print(f"- Half-life = {half_life:.2f} seconds ({half_life/3600:.4f} hours)")
            else:
                print("- Half-life: N/A (k=0)")
        elif reaction_order == 0:
            integrated_law_text_desc = f"[{reactant_names_for_kinetics[0]}] = [{reactant_names_for_kinetics[0]}]₀ - kt"
        elif reaction_order == 2: # Assumes 2A -> P or pseudo-second order
            integrated_law_text_desc = f"1/[{reactant_names_for_kinetics[0]}] = 1/[{reactant_names_for_kinetics[0]}]₀ + kt"
        else:
            integrated_law_text_desc = "Numerical integration used (see plot for details)"
        print(f"- Integrated rate law form: {integrated_law_text_desc}")


        # Simulate reaction progress over the user-defined simulation time
        time_points = np.linspace(0, self.simulation_time, 500)  # 500 points for smoother curves
        initial_conc = 1.0  # Starting with 1 mol/L (for the reactant whose kinetics are being modeled)
        reactant_conc, product_conc_total = self.simulate_reaction(initial_conc, k, time_points, reaction_order)

        print(f"\nSimulating reaction over {self.simulation_time/3600:.2f} hours ({self.simulation_time:.0f} seconds)...")
        print(f"- Initial concentration of {reactant_names_for_kinetics[0]} = {initial_conc:.2f} mol/L")
        final_reactant_conc_val = reactant_conc[-1] if reactant_conc.size > 0 else initial_conc
        print(f"- Final concentration of {reactant_names_for_kinetics[0]} = {final_reactant_conc_val:.4e} mol/L") # Use scientific notation for very small numbers
        conversion = (1 - final_reactant_conc_val/initial_conc) * 100 if initial_conc > 0 else 0
        print(f"- Conversion of {reactant_names_for_kinetics[0]} = {conversion:.1f}%")


        # Calculate conversion at various time points
        # Display conversion at the end of the simulation first
        end_sim_hours = self.simulation_time/3600
        print(f"- Conversion at {end_sim_hours:.2f} hours (end of simulation) = {conversion:.1f}%")

        # Then other checkpoints if they are within the simulation time
        conversion_checkpoints_hours = [0.1, 0.5, 1, 6, 12, 24]
        for hour_chk in conversion_checkpoints_hours:
            sec_chk = hour_chk * 3600
            if sec_chk < self.simulation_time: # Only print if checkpoint is before end of simulation
                idx = np.searchsorted(time_points, sec_chk, side='right') -1
                idx = max(0, idx) # Ensure index is not negative
                if idx < len(reactant_conc):
                    conv_chk = (1 - reactant_conc[idx]/initial_conc) * 100 if initial_conc > 0 else 0
                    print(f"- Conversion at {hour_chk:.1f} hour(s) = {conv_chk:.1f}%")


        # Calculate current reaction rate at the end of simulation
        # Based on the concentration of the primary reactant
        reactant_concs_at_end_for_rate = [final_reactant_conc_val]
        # If 2nd order and two reactants were specified, this simplification might not be fully accurate
        # but matches the pseudo-order assumption.
        current_rate = self.calculate_reaction_rate(k, reactant_concs_at_end_for_rate, reaction_order)
        print(f"\nCurrent reaction rate (at {end_sim_hours:.2f} hours, based on [{reactant_names_for_kinetics[0]}]) = {current_rate:.4e} mol/(L·s)")

        # Print predicted chemical transformation
        initial_reactant_concs_nominal = [initial_conc] * len(reactant_names_for_kinetics)
        self.print_stoichiometric_equation(reactant_names_for_kinetics, product_display_names,
                                           initial_reactant_concs_nominal, product_conc_total[-1] if product_conc_total.size > 0 else 0)

        # Plot reaction progress
        print("\nGenerating reaction progress plot...")
        self.plot_reaction_progress(time_points, reactant_conc, product_conc_total,
                                   reactant_names_for_kinetics, products_smiles, temperature,
                                   condition, k, reaction_order)

        # Visualize molecules
        print("\nGenerating molecular visualization...")
        self.visualize_molecules(reactants_smiles, products_smiles) # Uses all valid reactant SMILES

        # Display final summary
        print("\n*** Reaction Summary ***")
        print(f"Reactants considered for prediction: {' + '.join(reactant_names)}") # Original full list
        print(f"Reactants used for kinetics ({reactant_names_for_kinetics[0]} as primary): {' + '.join(reactant_names_for_kinetics)}")
        print(f"Products ({product_source}): {' + '.join(product_display_names)}")
        print(f"Reaction conditions: {condition} at {temperature}K")
        print(f"Rate constant (k): {k:.2e}")
        print(f"Reaction order (for {reactant_names_for_kinetics[0]} or pseudo-order): {reaction_order}")
        print(f"Current reaction rate (at {end_sim_hours:.2f} hrs): {current_rate:.4e} mol/(L·s)")

        # Re-iterate rate law for summary from the print_rate_law function's output logic
        summary_rate_law_text = self.print_rate_law(reactant_names_for_kinetics, reaction_order, k).replace("\nRate Law:\n","") # Get the string
        print(f"Rate Law (model): {summary_rate_law_text.split('= ')[0].strip()} = {summary_rate_law_text.split('= ')[1].strip()}") # Reformat slightly
        print(f"Integrated rate law form (model for {reactant_names_for_kinetics[0]}): {integrated_law_text_desc}")
        print(f"Conversion of {reactant_names_for_kinetics[0]} after {end_sim_hours:.2f} hours: {conversion:.1f}%")
        print("\nMolecular structures and reaction progress plots have been generated if possible.")


# --- Keep the main function as is ---
def main():
    """Main function to run the program interactively with optional product input"""
    predictor = ReactionPredictor()

    # Check if Hugging Face T5 model loaded successfully
    if predictor.hf_model is None:
        print("\nWarning: Hugging Face T5 model failed to load. Product prediction via model will not be available.")
        print("You must provide product names or SMILES manually for each reaction.")


    print("\n=== Chemical Reaction Prediction and Simulation System ===")
    print("This program predicts reaction pathways (via Hugging Face T5) or uses provided products,")
    print("and simulates reaction kinetics.")
    print("Type 'done' at any prompt to exit the program.\n")

    while True:
        # Get user input for reactants
        reactant_input = input("\nEnter reactant names (separated by comma), or 'done' to exit: ").strip()
        if reactant_input.lower() == 'done':
            print("Exiting program. Thank you!")
            break
        if not reactant_input:
            print("No reactants entered. Please try again.")
            continue

        reactant_names = [name.strip() for name in reactant_input.split(",") if name.strip()]
        if not reactant_names:
            print("No valid reactant names parsed. Please try again.")
            continue


        # Get optional user input for products
        product_prompt = "\nEnter product names or SMILES (separated by comma, optional - press Enter to predict with Hugging Face T5), or 'done' to exit: "
        if predictor.hf_model is None:
             product_prompt = "\nEnter product names or SMILES (separated by comma, REQUIRED as Hugging Face T5 model is not loaded), or 'done' to exit: "

        product_input = input(product_prompt).strip()
        if product_input.lower() == 'done':
            print("Exiting program. Thank you!")
            break
        if predictor.hf_model is None and not product_input:
            print("Hugging Face T5 model is not loaded, and no products were provided. Cannot proceed.")
            continue


        # Get reaction conditions
        print("\nAvailable conditions:")
        for condition_key in predictor.condition_params.keys():
            print(f"- {condition_key}")
        condition_input = input(f"\nEnter reaction condition (default: ideal), or 'done' to exit: ").strip().lower()
        if condition_input == 'done':
            print("Exiting program. Thank you!")
            break

        condition = condition_input if condition_input else "ideal"
        if condition not in predictor.condition_params:
            print(f"Invalid condition '{condition}'. Using 'ideal' as default.")
            condition = "ideal"

        # Get temperature
        temp_input = input("\nEnter temperature in Kelvin (default: 298.15 K), or 'done' to exit: ").strip()
        if temp_input.lower() == 'done':
            print("Exiting program. Thank you!")
            break

        try:
            temperature = float(temp_input) if temp_input else 298.15
            if temperature <= 0:
                raise ValueError("Temperature must be positive")
        except ValueError:
            print("Invalid temperature. Using 298.15 K as default.")
            temperature = 298.15

        # Get simulation time
        time_input_prompt = "\nEnter simulation time in seconds (e.g., 3600 for 1 hour; default: 86400 for 24 hours), or 'done' to exit: "
        time_input = input(time_input_prompt).strip()
        if time_input.lower() == 'done':
            print("Exiting program. Thank you!")
            break

        try:
            simulation_time = float(time_input) if time_input else 86400.0
            if simulation_time <= 0:
                print("Simulation time must be positive. Using default 86400 seconds (24 hours).")
                simulation_time = 86400.0
        except ValueError:
            print("Invalid simulation time. Using default 86400 seconds (24 hours).")
            simulation_time = 86400.0


        # Ask if user wants to provide a custom k value
        k_input = input("\nEnter a custom rate constant k(T) value (press Enter to calculate automatically), or 'done' to exit: ").strip()
        if k_input.lower() == 'done':
            print("Exiting program. Thank you!")
            break

        custom_k = None
        if k_input:
            try:
                custom_k_val = float(k_input)
                if custom_k_val < 0: # Rate constant cannot be negative
                    print("Rate constant must be non-negative. Will calculate automatically.")
                else:
                    custom_k = custom_k_val
            except ValueError:
                print("Invalid rate constant format. Will calculate automatically.")

        # Run the prediction and simulation, passing the optional product input
        predictor.run_prediction_and_simulation(
            reactant_names,
            condition,
            temperature,
            simulation_time,
            custom_k,
            user_product_input=product_input
        )

        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()