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
# from gradio_client import Client # Comment out or remove if no longer using Gradio

# --- Dependencies for T5Chem ---
# You will need to install these libraries:
# pip install torch # or tensorflow, depending on your setup and T5Chem build
# pip install transformers
# pip install t5chem # If you installed via pip
# or if installed from source, ensure the t5chem package is in your Python path
from transformers import T5ForConditionalGeneration
from t5chem import SimpleTokenizer # Assuming SimpleTokenizer is part of the t5chem package
import torch # Assuming PyTorch backend as suggested by the T5ForConditionalGeneration.from_pretrained example

# -----------------------------


class ReactionPredictor:
    def __init__(self):
        # --- Initialize the T5Chem model and tokenizer ---
        self.tokenizer = None
        self.model = None
        self.load_t5chem_model()
        # -----------------------------------------------------

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

    def load_t5chem_model(self):
        """Loads the T5Chem model and tokenizer."""
        print("Loading T5Chem model and tokenizer...")
        try:
            # --- INSERT ACTUAL PATH TO YOUR DOWNLOADED PRE-TRAINED MODEL ---
            # Replace "path/to/your/pretrained/model/" with the actual path
            # where you downloaded the T5Chem model files (including vocab.txt).
            pretrain_path = "c:\\Users\\shaan\\Documents\\School\\CSUDH\\Biocharge\\ChemPy, DeepChem\\New Code\\models\\pretrain\\simple\\" # Example path, update as needed
            
            # -------------------------------------------------------------

            if not os.path.exists(pretrain_path):
                print(f"Error: Pre-trained model path not found at {pretrain_path}")
                print("Please update 'pretrain_path' in load_t5chem_model with the correct location of your downloaded T5Chem model files.")
                return # Exit if path is incorrect

            # Load the model for sequence-to-sequence tasks (reaction prediction)
            self.model = T5ForConditionalGeneration.from_pretrained(pretrain_path)
            self.model.eval() # Set model to evaluation mode (important for inference)

            # Load the tokenizer using the vocab file from the model path
            vocab_file_path = os.path.join(pretrain_path, 'vocab.txt')
            if not os.path.exists(vocab_file_path):
                 print(f"Error: vocab.txt not found in the model path: {vocab_file_path}")
                 print("Please ensure the downloaded model files include vocab.txt and the 'pretrain_path' is correct.")
                 self.model = None # Invalidate model if tokenizer cannot be loaded
                 return

            self.tokenizer = SimpleTokenizer(vocab_file=vocab_file_path)


            print("T5Chem model and tokenizer loaded successfully.")

        except Exception as e:
            print(f"Error loading T5Chem model: {e}")
            self.tokenizer = None
            self.model = None # Ensure model is None if loading fails


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


    def predict_reaction(self, reactants_smiles):
        """Predict reaction products using the loaded T5Chem model."""
        if self.model is None or self.tokenizer is None:
            print("T5Chem model or tokenizer not loaded. Cannot perform prediction.")
            # Fallback to returning the first reactant as product
            return [reactants_smiles[0]]

        # --- Use the T5Chem API based on the documentation snippet ---
        reactants_string = ".".join(reactants_smiles)
        # Add the "Product:" prompt as shown in the documentation
        input_text = f"Product:{reactants_string}>>" # Note the '>>' might be part of the expected input format

        print(f"Querying T5Chem model with input: {input_text}")

        try:
            # Encode the input text
            # Ensure return_tensors is set correctly for your backend ('pt' for PyTorch)
            inputs = self.tokenizer.encode(input_text, return_tensors='pt')

            # Move inputs to the same device as the model if using GPU
            if torch.cuda.is_available():
                 inputs = inputs.to(self.model.device)

            # Generate prediction
            # You might need to tune generation parameters like num_beams, max_length, etc.
            # The documentation snippet uses max_length=300 and early_stopping=True
            with torch.no_grad(): # Use torch.no_grad() for inference to save memory and speed up computation
                 output_ids = self.model.generate(
                     inputs,
                     max_length=512, # Increased max_length slightly for potentially longer products
                     early_stopping=True,
                     num_beams=5 # Adding num_beams can improve prediction quality
                 )

            # Decode the generated output IDs
            # The documentation snippet uses output[0] and skip_special_tokens=True
            predicted_smiles_string = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # T5Chem outputs product SMILES. Parse the string to get a list of individual products.
            # The documentation example shows a single product output.
            # If multiple products are possible, they might be separated by '.'.
            products_smiles = [s.strip() for s in predicted_smiles_string.split('.') if s.strip()]

            print(f"T5Chem predicted raw output: {predicted_smiles_string}")
            print(f"Parsed product SMILES: {products_smiles}")

            # Validate and filter predicted products (keep your existing validation logic)
            valid_products = []
            for smiles in products_smiles:
                 try:
                     mol = Chem.MolFromSmiles(smiles)
                     if mol:
                         valid_products.append(smiles)
                 except:
                     print(f"Invalid SMILES predicted by T5Chem: '{smiles}'. Skipping.")
                     continue

            if not valid_products:
                 print("No valid product SMILES found from T5Chem prediction. Generating default product.")
                 # Fallback to returning the first reactant as product
                 valid_products = [reactants_smiles[0]]

            return valid_products

        except Exception as e:
            print(f"Error during T5Chem reaction prediction: {e}")
            # Fallback to returning the first reactant as product
            return [reactants_smiles[0]]
        # ---------------------------------------------------------


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
            return k * concentrations[0]**order

    def print_stoichiometric_equation(self, reactant_names, product_names, reactant_concs, product_concs):
        """Print the stoichiometric equation with molarities"""
        reactant_terms = []
        for i, name in enumerate(reactant_names):
            reactant_terms.append(f"{reactant_concs[i]:.2f} M {name}")

        product_terms = []
        for i, name in enumerate(product_names):
            product_terms.append(f"{product_concs[i]:.2f} M {name}")

        equation = " + ".join(reactant_terms) + " → " + " + ".join(product_terms)
        print("\nStoichiometric Equation:")
        print(equation)
        return equation

    def print_rate_law(self, reactant_names, order, k):
        """Print the rate law equation"""
        if order == 0:
            rate_law = f"Rate = k = {k:.2e} mol/(L·s)"
        elif order == 1:
            rate_law = f"Rate = k[{reactant_names[0]}] = {k:.2e}[{reactant_names[0]}] s⁻¹"
        elif order == 2:
            if len(reactant_names) > 1:
                rate_law = f"Rate = k[{reactant_names[0]}][{reactant_names[1]}] = {k:.2e}[{reactant_names[0]}][{reactant_names[1]}] L/(mol·s)"
            else:
                rate_law = f"Rate = k[{reactant_names[0]}]² = {k:.2e}[{reactant_names[0]}]² L/(mol·s)"
        else:
            rate_law = f"Rate = k[{reactant_names[0]}]^{order} = {k:.2e}[{reactant_names[0]}^{order}"

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
            # Second-order: 1/[A] = 1/[A]₀ + kt
            concentrations = initial_conc / (1 + initial_conc * k * time_points)

        else:
            # For other orders, use numerical integration
            def reaction_rate(t, C):
                return -k * (C ** order)

            solution = solve_ivp(
                reaction_rate,
                [time_points[0], time_points[-1]],
                [initial_conc],
                t_eval=time_points
            )

            concentrations = solution.y[0]

        return concentrations


    def simulate_reaction(self, initial_conc, k, time_points, order):
        """Simulate reaction progress over time using integrated rate law"""
        reactant_conc = self.integrated_rate_law(order, initial_conc, k, time_points)

        # Calculate product concentration based on stoichiometry (assuming 1:1)
        product_conc = initial_conc - reactant_conc

        return reactant_conc, product_conc


    def plot_reaction_progress(self, time_points, reactant_conc, product_conc, reactant_names, products, temperature, condition, k, order):
        """Plot the reaction progress over time"""
        plt.figure(figsize=(12, 8))

        # Plot reactant concentration decrease
        plt.plot(time_points, reactant_conc, 'b-', linewidth=2, label="Reactants")

        # Plot product formation
        plt.plot(time_points, product_conc, 'r-', linewidth=2, label="Products")

        # Calculate half-life if applicable
        if order == 1:
            half_life = np.log(2) / k
            if half_life <= time_points[-1]:
                plt.axvline(x=half_life, color='g', linestyle='--', label=f"Half-life = {half_life:.2f}s")

        # Format time axis with appropriate units
        if time_points[-1] > 3600:
            plt.xlabel('Time (hours)', fontsize=12)
            time_hours = time_points / 3600
            plt.xticks(np.linspace(0, time_points[-1], 13),
                       [f"{h:.1f}" for h in np.linspace(0, time_hours[-1], 13)])
        else:
            plt.xlabel('Time (s)', fontsize=12)

        plt.ylabel('Concentration (mol/L)', fontsize=12)
        reaction_time_text = f"{time_points[-1]/3600:.1f} hours" if time_points[-1] > 3600 else f"{time_points[-1]:.0f} seconds"
        plt.title(f'Reaction Progress: {" + ".join(reactant_names)} at {temperature}K ({condition}) over {reaction_time_text}\n' +
                 f'Rate constant k = {k:.2e}, Reaction order = {order}', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)

        # Add rate law equation to the plot
        if order == 0:
            rate_law = f"Rate = k = {k:.2e}"
            integrated_law = f"[A] = [A]₀ - kt"
        elif order == 1:
            rate_law = f"Rate = k[A] where k = {k:.2e}"
            integrated_law = f"[A] = [A]₀e^(-kt)"
        elif order == 2:
            rate_law = f"Rate = k[A]² where k = {k:.2e}"
            integrated_law = f"1/[A] = 1/[A]₀ + kt"
        else:
            rate_law = f"Rate = k[A]^{order} where k = {k:.2e}"
            integrated_law = f"Numerical integration used"

        plt.figtext(0.5, 0.01, f"Rate Law: {rate_law}\nIntegrated Form: {integrated_law}",
                   ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})

        # Save the plot to a temporary file and return the filename
        plt.tight_layout()
        filename = "reaction_progress.png"
        plt.savefig(filename)
        plt.show()
        return filename

    def visualize_molecules(self, reactants_smiles, products_smiles):
        """Create visual representations of reactants and products"""
        reactant_mols = [Chem.MolFromSmiles(smiles) for smiles in reactants_smiles if smiles]
        product_mols = [Chem.MolFromSmiles(smiles) for smiles in products_smiles if smiles]

        # Filter out None values (failed SMILES conversions)
        reactant_mols = [mol for mol in reactant_mols if mol is not None]
        product_mols = [mol for mol in product_mols if mol is not None]

        if not reactant_mols or not product_mols:
            print("Warning: Could not visualize some molecules due to invalid SMILES.")
            return None

        # Generate 2D coordinates for better visualization
        for mol in reactant_mols + product_mols:
            if mol: # Check if molecule is valid before generating coordinates
                AllChem.Compute2DCoords(mol)

        # Create an image showing the reaction
        try:
            # Check if both reactant and product molecules exist before attempting ReactionToImage
            if reactant_mols and product_mols:
                img = Draw.ReactionToImage(reactant_mols, product_mols, subImgSize=(300, 300))

                # Save the image
                filename = "reaction_visualization.png"
                if img:
                    img.save(filename)
                    return filename
            else:
                 print("Not enough valid reactant or product molecules for reaction visualization. Attempting individual molecule visualization.")
                 # Fallback to individual molecule visualization if reaction visualization fails

        except Exception as e:
            print(f"Failed to create reaction visualization: {e}")
            print("Attempting individual molecule visualization.")

        # Alternative: create separate images if reaction visualization fails or is not possible
        try:
            reactant_img = Draw.MolsToGridImage(reactant_mols, molsPerRow=2, subImgSize=(200, 200),
                                               legends=[f"Reactant {i+1}" for i in range(len(reactant_mols))])
            product_img = Draw.MolsToGridImage(product_mols, molsPerRow=2, subImgSize=(200, 200),
                                             legends=[f"Product {i+1}" for i in range(len(product_mols))])

            reactant_img.save("reactants.png")
            product_img.save("products.png")
            return "reactants.png" # Return the reactant image filename as a representative
        except Exception as e:
            print(f"Failed to create molecule visualization: {e}")


        return None # Return None if all visualization attempts fail


    def run_prediction_and_simulation(self, reactant_names, condition, temperature, custom_k=None):
        """Main method to run the entire prediction and simulation workflow"""
        print(f"\n*** Starting reaction prediction for {' + '.join(reactant_names)} under {condition} conditions at {temperature}K ***\n")

        # Convert common names to SMILES
        reactants_smiles = []
        for name in reactant_names:
            smiles = self.get_smiles_from_pubchem(name)
            if smiles:
                reactants_smiles.append(smiles)
                print(f"Retrieved SMILES for {name}: {smiles}")
            else:
                print(f"Failed to retrieve SMILES for {name}")

        if len(reactants_smiles) < 1:
            print("Error: Need at least one valid reactant with SMILES notation.")
            return

        # Predict reaction products
        print("\nPredicting reaction products...")
        products_smiles = self.predict_reaction(reactants_smiles)
        print(f"Predicted products SMILES: {products_smiles}")

        # Get product names (using molecular formula as placeholder)
        product_names = []
        for i, smiles in enumerate(products_smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
                    product_names.append(f"Product_{i+1} ({formula})")
                else:
                    product_names.append(f"Product_{i+1}")
            except:
                product_names.append(f"Product_{i+1}")

        # Calculate reaction rate constant
        if custom_k is not None:
            k, reaction_order = self.use_custom_k(custom_k, condition)
            print(f"\nUsing custom rate constant k = {k:.2e}")
        else:
            k, reaction_order = self.calculate_rate_constant(temperature, condition)
            print(f"\nCalculated rate constant:")
            print(f"- Rate constant k = {k:.2e}")

        print(f"- Reaction order = {reaction_order}")

        # Print rate law equation
        self.print_rate_law(reactant_names, reaction_order, k)

        # Determine the applicable integrated rate law
        if reaction_order == 0:
            rate_law = "[A] = [A]₀ - kt"
        elif reaction_order == 1:
            rate_law = "[A] = [A]₀e^(-kt)"
            half_life = np.log(2) / k
            print(f"- Half-life = {half_life:.2f} seconds ({half_life/3600:.4f} hours)")
        elif reaction_order == 2:
            rate_law = "1/[A] = 1/[A]₀ + kt"
        else:
            rate_law = "Numerical integration used"
        print(f"- Integrated rate law: {rate_law}")

        # Simulate reaction progress over the user-defined simulation time
        time_points = np.linspace(0, self.simulation_time, 200)  # 200 points for smoother curves
        initial_conc = 1.0  # Starting with 1 mol/L
        reactant_conc, product_conc = self.simulate_reaction(initial_conc, k, time_points, reaction_order)

        print(f"\nSimulating reaction over {self.simulation_time/3600:.1f} hours ({self.simulation_time} seconds)...")
        print(f"- Initial concentration = {initial_conc} mol/L")
        print(f"- Final reactant concentration = {reactant_conc[-1]:.4f} mol/L")
        print(f"- Final product concentration = {product_conc[-1]:.4f} mol/L")
        print(f"- Conversion = {(1 - reactant_conc[-1]/initial_conc) * 100:.1f}%")

        # Calculate conversion at various time points (adjusting for potentially shorter simulation time)
        conversion_checkpoints_hours = [1, 6, 12, 24]
        for hour in conversion_checkpoints_hours:
            sec = hour * 3600
            if sec > self.simulation_time:
                # Check conversion at the end of simulation if it's before the standard checkpoint
                if hour == conversion_checkpoints_hours[0] and self.simulation_time > 0:
                     # Use the last point if simulation is shorter than the first checkpoint
                     conv = (1 - reactant_conc[-1]/initial_conc) * 100
                     print(f"- Conversion at {self.simulation_time/3600:.1f} hours (end of simulation) = {conv:.1f}%")
                break # Stop checking if the checkpoint is beyond simulation time

            idx = np.searchsorted(time_points, sec)
            if idx < len(reactant_conc):
                conv = (1 - reactant_conc[idx]/initial_conc) * 100
                print(f"- Conversion at {hour} hour(s) = {conv:.1f}%")


        # Calculate current reaction rate at the end of simulation
        reactant_concs_at_end = [reactant_conc[-1]]
        current_rate = self.calculate_reaction_rate(k, reactant_concs_at_end, reaction_order)
        print(f"\nCurrent reaction rate (at {self.simulation_time/3600:.1f} hours) = {current_rate:.4e} mol/(L·s)")

        # Print stoichiometric equation
        initial_reactant_concs = [initial_conc] * len(reactant_names)
        # For product concentrations in the stoichiometry, it's more representative to use final concentrations
        final_product_concs = [product_conc[-1]] * len(product_names)
        self.print_stoichiometric_equation(reactant_names, product_names, initial_reactant_concs, final_product_concs)

        # Plot reaction progress
        print("\nGenerating reaction progress plot...")
        self.plot_reaction_progress(time_points, reactant_conc, product_conc,
                                   reactant_names, products_smiles, temperature,
                                   condition, k, reaction_order)

        # Visualize molecules
        print("\nGenerating molecular visualization...")
        molecule_viz = self.visualize_molecules(reactants_smiles, products_smiles)

        # Display final summary
        print("\n*** Reaction Summary ***")
        print(f"Reactants: {' + '.join(reactant_names)}")
        print(f"Products: {' + '.join(product_names)}")
        print(f"Reaction conditions: {condition} at {temperature}K")
        print(f"Rate constant: k = {k:.2e}")
        print(f"Current reaction rate: {current_rate:.4e} mol/(L·s)")
        # Reprinting rate law here for summary cohesion
        if reaction_order == 0:
            summary_rate_law = f"Rate = k = {k:.2e} mol/(L·s)"
            summary_integrated_law = f"[A] = [A]₀ - kt"
        elif reaction_order == 1:
            summary_rate_law = f"Rate = k[A] where k = {k:.2e}"
            summary_integrated_law = f"[A] = [A]₀e^(-kt)"
        elif reaction_order == 2:
            summary_rate_law = f"Rate = k[A]² where k = {k:.2e}"
            summary_integrated_law = f"1/[A] = 1/[A]₀ + kt"
        else:
            summary_rate_law = f"Rate = k[A]^{reaction_order} where k = {k:.2e}"
            summary_integrated_law = f"Numerical integration used"

        print(f"Rate law: {summary_rate_law}")
        print(f"Integrated rate law: {summary_integrated_law}")
        print(f"Conversion after {self.simulation_time/3600:.1f} hours: {(1 - reactant_conc[-1]/initial_conc) * 100:.1f}%")
        print("\nMolecular structures and reaction progress have been plotted.")


# --- Keep the main function as is ---
def main():
    """Main function to run the program interactively"""
    predictor = ReactionPredictor()

    print("=== Chemical Reaction Prediction and Simulation System ===")
    print("This program predicts reaction pathways and simulates reaction kinetics.")
    print("Type 'done' at any prompt to exit the program.\n")

    while True:
        # Get user input for reactants
        reactant_input = input("\nEnter reactant names (separated by comma), or 'done' to exit: ").strip()
        if reactant_input.lower() == 'done':
            print("Exiting program. Thank you!")
            break

        reactant_names = [name.strip() for name in reactant_input.split(",")]

        # Get reaction conditions
        print("\nAvailable conditions:")
        for condition in predictor.condition_params.keys():
            print(f"- {condition}")
        condition_input = input("\nEnter reaction condition, or 'done' to exit: ").strip().lower()
        if condition_input == 'done':
            print("Exiting program. Thank you!")
            break

        condition = condition_input
        if condition not in predictor.condition_params:
            print(f"Invalid condition. Using 'ideal' as default.")
            condition = "ideal"

        # Get temperature
        temp_input = input("\nEnter temperature (K), or 'done' to exit: ").strip()
        if temp_input.lower() == 'done':
            print("Exiting program. Thank you!")
            break

        try:
            temperature = float(temp_input)
            if temperature <= 0:
                raise ValueError("Temperature must be positive")
        except ValueError:
            print("Invalid temperature. Using 298K as default.")
            temperature = 298.0

        # Get simulation time
        time_input = input("\nEnter simulation time in seconds (default: 86400 for 24 hours), or 'done' to exit: ").strip()
        if time_input.lower() == 'done':
            print("Exiting program. Thank you!")
            break

        try:
            simulation_time = float(time_input)
            if simulation_time <= 0:
                print("Simulation time must be positive. Using default 86400 seconds (24 hours).")
                simulation_time = 86400 # Reverted to 24 hours default
        except ValueError:
            print("Invalid simulation time. Using default 86400 seconds (24 hours).")
            simulation_time = 86400 # Reverted to 24 hours default

        # Set the simulation time in the predictor instance
        predictor.simulation_time = simulation_time

        # Ask if user wants to provide a custom k value
        k_input = input("\nEnter a custom rate constant k(T) value (press Enter to calculate automatically), or 'done' to exit: ").strip()
        if k_input.lower() == 'done':
            print("Exiting program. Thank you!")
            break

        custom_k = None
        if k_input:
            try:
                custom_k = float(k_input)
                if custom_k < 0:
                    print("Rate constant must be non-negative. Will calculate automatically.")
                    custom_k = None
            except ValueError:
                print("Invalid rate constant. Will calculate automatically.")

        # Run the prediction and simulation
        predictor.run_prediction_and_simulation(reactant_names, condition, temperature, custom_k)

        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()