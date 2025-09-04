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

# --- Dependencies for DeepChem ---
# You will need to install DeepChem and its dependencies:
# pip install deepchem
# DeepChem often uses TensorFlow or PyTorch as backend.
# pip install tensorflow # or pip install torch
import deepchem as dc
# ---------------------------------


class ReactionPredictor:
    def __init__(self):
        # --- Initialize the AI Model (DeepChem) ---
        self.deepchem_model = None
        self.load_deepchem_model() # Call the method to load the DeepChem model
        # ------------------------------------------

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

    def load_deepchem_model(self):
        """Loads the chosen DeepChem reaction prediction model."""
        print("Loading DeepChem model...")
        try:
            # --- PLACEHOLDER FOR ACTUAL DEEPCHEM MODEL LOADING CODE ---
            # This code depends heavily on the specific DeepChem model you are using
            # and how its weights were saved.
            #
            # Example (Illustrative - replace with your model's loading logic):
            # model_path = "path/to/your/deepchem_model_checkpoint" # Replace with actual path
            # if not os.path.exists(model_path):
            #     print(f"Error: DeepChem model path not found at {model_path}")
            #     print("Please update 'model_path' in load_deepchem_model with the correct location.")
            #     self.deepchem_model = None
            #     return
            #
            # # Assuming a GraphConvModel for reaction prediction (example)
            # # You would need the correct model class and parameters
            # self.deepchem_model = dc.models.GraphConvModel(...) # Replace with your model class and params
            # self.deepchem_model.restore(model_path) # Or use a different loading method
            # print("DeepChem model loaded successfully.")
            # ---------------------------------------------------------

            print("DeepChem model loading placeholder executed. Replace with actual loading code.")
            self.deepchem_model = True # Set to a non-None value to indicate 'loaded' (placeholder)

        except Exception as e:
            print(f"Error loading DeepChem model: {e}")
            self.deepchem_model = None # Ensure model is None if loading fails


    # --- Removed T5Chem and Gemini API setup methods ---
    # def setup_gemini_api(self):
    #     pass # Removed Gemini API setup

    # def load_t5chem_model(self):
    #     pass # Removed T5Chem loading
    # ---------------------------------------------------


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


    def predict_reaction_deepchem(self, reactants_smiles):
        """Predict reaction products using the DeepChem model."""
        if self.deepchem_model is None:
            print("DeepChem model not loaded. Cannot perform prediction.")
            # Fallback to returning the first reactant as product
            return [reactants_smiles[0]]

        print("Querying DeepChem model...")

        try:
            # --- PLACEHOLDER FOR ACTUAL DEEPCHEM PREDICTION CODE ---
            # This code depends heavily on the specific DeepChem model you are using
            # and its expected input/output format.
            #
            # 1. Featurize reactants: Convert SMILES to the model's input format.
            # featurizer = dc.feat.GraphConvFeaturizer() # Example featurizer
            # reactant_mols = [Chem.MolFromSmiles(s) for s in reactants_smiles if s]
            # features = featurizer.featurize(reactant_mols)
            #
            # 2. Create a DeepChem Dataset (often required for prediction)
            # dataset = dc.data.NumpyDataset(X=features)
            #
            # 3. Make prediction using the loaded model.
            # The output format will depend on the model. If it's a generative model,
            # it should output product SMILES.
            # predicted_output = self.deepchem_model.predict(dataset)
            #
            # 4. Parse the model's output to get product SMILES strings.
            # This parsing logic will be specific to your model's output format.
            # products_smiles = self.parse_deepchem_output(predicted_output) # You need to implement this helper

            # --- Placeholder output ---
            print("DeepChem prediction placeholder executed. Replace with actual prediction code.")
            # For now, return the first reactant as a placeholder product
            products_smiles = [reactants_smiles[0]]
            # ---------------------------------------------------------

            # Validate and filter predicted products
            valid_products = []
            for smiles in products_smiles:
                 try:
                     mol = Chem.MolFromSmiles(smiles)
                     if mol:
                         valid_products.append(smiles)
                     else:
                         print(f"Invalid SMILES predicted by DeepChem: '{smiles}'. Skipping.")
                 except:
                     print(f"Error parsing SMILES predicted by DeepChem: '{smiles}'. Skipping.")
                     continue

            if not valid_products:
                 print("No valid product SMILES found from DeepChem prediction. Generating default product.")
                 # Fallback to returning the first reactant as product
                 valid_products = [reactants_smiles[0]]

            return valid_products

        except Exception as e:
            print(f"Error during DeepChem reaction prediction: {e}")
            # Fallback to returning the first reactant as product
            return [reactants_smiles[0]]

    # You would likely need a helper method to parse the model's output
    # def parse_deepchem_output(self, model_output):
    #     """Parses the output of the DeepChem reaction prediction model."""
    #     # Implement parsing logic based on your chosen DeepChem model's output format
    #     pass


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

        if not reactant_mols and not product_mols: # Changed from or to and, if both are empty, nothing to visualize
            print("Warning: Could not visualize any molecules.")
            return None

        # Generate 2D coordinates for better visualization
        for mol in reactant_mols + product_mols:
            if mol: # Check if molecule is valid before generating coordinates
                AllChem.Compute2DCoords(mol)

        # --- Removed the incorrect Draw.ReactionToImage call ---
        # The fallback to MolsToGridImage will now always be used.
        try:
            # Alternative: create separate images
            reactant_img = Draw.MolsToGridImage(reactant_mols, molsPerRow=4, subImgSize=(200, 200), # Increased molsPerRow for potentially better layout
                                               legends=[f"Reactant {i+1}" for i in range(len(reactant_mols))])
            product_img = Draw.MolsToGridImage(product_mols, molsPerRow=4, subImgSize=(200, 200), # Increased molsPerRow
                                             legends=[f"Product {i+1}" for i in range(len(product_mols))])

            # Save both images
            reactant_filename = "reactants.png"
            product_filename = "products.png"
            if reactant_img:
                 reactant_img.save(reactant_filename)
                 print(f"Saved reactant visualization to {reactant_filename}")
            if product_img:
                 product_img.save(product_filename)
                 print(f"Saved product visualization to {product_filename}")


            if reactant_img or product_img: # Return a filename if at least one image was saved
                 return reactant_filename # Return reactant filename as representative

        except Exception as e:
            print(f"Failed to create molecule visualization using MolsToGridImage: {e}")


        return None # Return None if all visualization attempts fail


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

        # --- Determine products: Use user input if provided, otherwise predict with DeepChem ---
        products_smiles = []
        product_source = "Predicted by DeepChem" # Default source

        if user_product_input and user_product_input.strip():
            print("\nUsing user-provided products...")
            user_product_names = [name.strip() for name in user_product_input.split(",")]
            for name in user_product_names:
                smiles = self.get_smiles_from_pubchem(name)
                if smiles:
                    products_smiles.append(smiles)
                    print(f"Retrieved SMILES for user-provided product {name}: {smiles}")
                else:
                    print(f"Failed to retrieve SMILES for user-provided product {name}. Skipping.")

            if not products_smiles:
                 print("Error: No valid SMILES found for user-provided products. Cannot proceed with kinetics.")
                 return # Exit if user provided products but none were valid

            product_source = "Provided by User"

        else:
            print("\nPredicting reaction products using DeepChem...")
            # Call the DeepChem prediction method
            products_smiles = self.predict_reaction_deepchem(reactants_smiles)

            if not products_smiles:
                print("DeepChem prediction failed or returned no valid products. Cannot proceed.")
                return # Exit if prediction failed

        print(f"Products SMILES ({product_source}): {products_smiles}")

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

        # --- Proceed with kinetics calculations and visualization using the determined products ---

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
        initial_conc = 1.0  # Starting with 1 mol/L (assuming 1:1 stoichiometry for kinetics)
        reactant_conc, product_conc = self.simulate_reaction(initial_conc, k, time_points, reaction_order)

        print(f"\nSimulating reaction over {self.simulation_time/3600:.1f} hours ({self.simulation_time} seconds)...")
        print(f"- Initial concentration = {initial_conc} mol/L")
        print(f"- Final reactant concentration = {reactant_conc[-1]:.4f} mol/L")
        conversion = (1 - reactant_conc[-1]/initial_conc) * 100
        print(f"- Conversion = {conversion:.1f}%")

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
        print(f"Products ({product_source}): {' + '.join(product_names)}") # Indicate source of products
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
        print(f"Conversion after {self.simulation_time/3600:.1f} hours: {conversion:.1f}%")
        print("\nMolecular structures and reaction progress have been plotted.")


# --- Keep the main function as is ---
def main():
    """Main function to run the program interactively with optional product input"""
    predictor = ReactionPredictor()

    # Check if DeepChem model loaded successfully
    if predictor.deepchem_model is None:
        print("\nWarning: DeepChem model failed to load. Product prediction will not be available.")
        print("You must provide product names manually for each reaction.")


    print("\n=== Chemical Reaction Prediction and Simulation System ===")
    print("This program predicts reaction pathways (via DeepChem) or uses provided products,")
    print("and simulates reaction kinetics.")
    print("Type 'done' at any prompt to exit the program.\n")

    while True:
        # Get user input for reactants
        reactant_input = input("\nEnter reactant names (separated by comma), or 'done' to exit: ").strip()
        if reactant_input.lower() == 'done':
            print("Exiting program. Thank you!")
            break

        reactant_names = [name.strip() for name in reactant_input.split(",")]

        # Get optional user input for products
        # Adjust prompt based on whether DeepChem model is available
        product_prompt = "\nEnter product names (separated by comma, optional - press Enter to predict with DeepChem), or 'done' to exit: "
        if predictor.deepchem_model is None:
             product_prompt = "\nEnter product names (separated by comma, REQUIRED if no DeepChem model - press Enter to skip), or 'done' to exit: "

        product_input = input(product_prompt).strip()
        if product_input.lower() == 'done':
            print("Exiting program. Thank you!")
            break

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

        # Run the prediction and simulation, passing the optional product input
        predictor.run_prediction_and_simulation(
            reactant_names,
            condition,
            temperature,
            simulation_time, # Pass simulation_time
            custom_k,
            user_product_input=product_input # Pass the user's product input string
        )

        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()


