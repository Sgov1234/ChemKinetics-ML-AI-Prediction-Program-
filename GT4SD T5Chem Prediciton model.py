#Models not loading and T5Chem not predcitng correct products.(fix other moels and try find new models or define organometallic/organic heuistrics rule)

import requests
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, rdMolDescriptors, Crippen, Lipinski
import time
import pandas as pd
import io
import os
from scipy.integrate import solve_ivp
import json
import urllib.parse
from typing import List, Dict, Optional, Tuple, Any
import re

# --- Dependencies for Hugging Face Transformers ---
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# --- Additional visualization libraries ---
try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False
    print("py3Dmol not available. 3D visualization will be limited.")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available. Enhanced image processing will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Interactive plots will be limited.")

try:
    import pubchempy as pcp
    PUBCHEMPY_AVAILABLE = True
except ImportError:
    PUBCHEMPY_AVAILABLE = False
    print("PubChemPy not available. Install with: pip install pubchempy")

try:
    import cirpy
    CIRPY_AVAILABLE = True
except ImportError:
    CIRPY_AVAILABLE = False
    print("CIRPy not available. Install with: pip install cirpy")


class ChemicalDatabaseInterface:
    """Enhanced interface for accessing multiple chemical databases"""
    
    def __init__(self):
        self.cache = {}  # Cache for previously retrieved SMILES
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ChemicalReactionPredictor/1.0 (Educational)'
        })
        
    def get_smiles(self, compound_name: str) -> Optional[str]:
        """
        Retrieve SMILES from multiple sources with fallback strategy
        """
        compound_name = compound_name.strip()
        
        # Check cache first
        if compound_name.lower() in self.cache:
            return self.cache[compound_name.lower()]
        
        # Check if it's already a SMILES string
        if self.is_valid_smiles(compound_name):
            self.cache[compound_name.lower()] = compound_name
            return compound_name
        
        smiles = None
        
        # Try different sources in order of reliability/speed
        sources = [
            ('PubChemPy', self.get_from_pubchempy),
            ('PubChem API', self.get_from_pubchem_api),
            ('CIR', self.get_from_cir),
            ('ChemSpider', self.get_from_chemspider_api),
            ('OPSIN', self.get_from_opsin),
            ('NCI CIR', self.get_from_nci_cir),
        ]
        
        for source_name, source_func in sources:
            try:
                print(f"Trying {source_name} for '{compound_name}'...")
                smiles = source_func(compound_name)
                if smiles:
                    print(f"✓ Found in {source_name}: {smiles}")
                    self.cache[compound_name.lower()] = smiles
                    return smiles
            except Exception as e:
                print(f"  {source_name} failed: {str(e)[:50]}")
                continue
        
        # Try common name variations
        smiles = self.try_name_variations(compound_name)
        if smiles:
            self.cache[compound_name.lower()] = smiles
            return smiles
            
        print(f"Could not find SMILES for '{compound_name}' in any database")
        return None
    
    def is_valid_smiles(self, text: str) -> bool:
        """Check if a string is a valid SMILES"""
        try:
            mol = Chem.MolFromSmiles(text)
            return mol is not None
        except:
            return False
    
    def get_from_pubchempy(self, compound_name: str) -> Optional[str]:
        """Get SMILES using PubChemPy library"""
        if not PUBCHEMPY_AVAILABLE:
            return None
        
        try:
            compounds = pcp.get_compounds(compound_name, 'name')
            if compounds:
                return compounds[0].canonical_smiles
        except:
            pass
        
        # Try synonym search
        try:
            compounds = pcp.get_compounds(compound_name, 'name', listkey_type='synonym')
            if compounds:
                return compounds[0].canonical_smiles
        except:
            pass
        
        return None
    
    def get_from_pubchem_api(self, compound_name: str) -> Optional[str]:
        """Get SMILES from PubChem REST API"""
        # Try exact name match
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(compound_name)}/property/CanonicalSMILES/JSON"
        
        try:
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "PropertyTable" in data and "Properties" in data["PropertyTable"]:
                    return data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
        except:
            pass
        
        # Try synonym search
        try:
            # First get CID from synonym
            synonym_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/synonym/{urllib.parse.quote(compound_name)}/cids/JSON"
            response = self.session.get(synonym_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "IdentifierList" in data and "CID" in data["IdentifierList"]:
                    cid = data["IdentifierList"]["CID"][0]
                    # Now get SMILES from CID
                    smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
                    response = self.session.get(smiles_url, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if "PropertyTable" in data and "Properties" in data["PropertyTable"]:
                            return data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
        except:
            pass
        
        return None
    
    def get_from_cir(self, compound_name: str) -> Optional[str]:
        """Get SMILES using CIRPy (Chemical Identifier Resolver)"""
        if not CIRPY_AVAILABLE:
            return None
        
        try:
            smiles = cirpy.resolve(compound_name, 'smiles')
            if smiles:
                return smiles
        except:
            pass
        
        return None
    
    def get_from_nci_cir(self, compound_name: str) -> Optional[str]:
        """Get SMILES from NCI Chemical Identifier Resolver"""
        url = f"https://cactus.nci.nih.gov/chemical/structure/{urllib.parse.quote(compound_name)}/smiles"
        
        try:
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                smiles = response.text.strip()
                if self.is_valid_smiles(smiles):
                    return smiles
        except:
            pass
        
        return None
    
    def get_from_opsin(self, compound_name: str) -> Optional[str]:
        """Get SMILES from OPSIN (Open Parser for Systematic IUPAC nomenclature)"""
        url = f"https://opsin.ch.cam.ac.uk/opsin/{urllib.parse.quote(compound_name)}.smi"
        
        try:
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                smiles = response.text.strip()
                if self.is_valid_smiles(smiles):
                    return smiles
        except:
            pass
        
        return None
    
    def get_from_chemspider_api(self, compound_name: str) -> Optional[str]:
        """Get SMILES from ChemSpider (requires API key for full access)"""
        # Basic search without API key (limited)
        url = f"https://www.chemspider.com/Search.asmx/SimpleSearch?query={urllib.parse.quote(compound_name)}"
        
        try:
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                # This would need parsing of the response
                # For now, returning None as it requires more complex handling
                pass
        except:
            pass
        
        return None
    
    def try_name_variations(self, compound_name: str) -> Optional[str]:
        """Try common name variations"""
        variations = []
        
        # Remove common prefixes/suffixes
        name = compound_name.lower()
        
        # Try without numbers at the beginning
        if re.match(r'^\d+[,-]', name):
            variations.append(re.sub(r'^\d+[,-]\s*', '', name))
        
        # Try with/without hyphens
        if '-' in name:
            variations.append(name.replace('-', ' '))
            variations.append(name.replace('-', ''))
        else:
            variations.append(name.replace(' ', '-'))
        
        # Try common alternative names
        replacements = {
            'sulphate': 'sulfate',
            'sulphide': 'sulfide',
            'sulphur': 'sulfur',
            'aluminium': 'aluminum',
            'aether': 'ether',
            'acetonitrile': 'methyl cyanide',
            'acetic acid': 'ethanoic acid',
            'formic acid': 'methanoic acid',
        }
        
        for old, new in replacements.items():
            if old in name:
                variations.append(name.replace(old, new))
            if new in name:
                variations.append(name.replace(new, old))
        
        # Try each variation
        for var in variations:
            if var != compound_name:
                print(f"  Trying variation: '{var}'")
                smiles = self.get_from_pubchem_api(var)
                if smiles:
                    return smiles
        
        return None


class OrganometallicReactionPredictor:
    """Specialized predictor for organometallic reactions"""
    
    def __init__(self):
        self.metal_patterns = {
            'transition_metals': ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                                 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                                 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'],
            'main_group_metals': ['Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba',
                                 'Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi'],
        }
        
        self.reaction_patterns = {
            'oxidative_addition': {
                'metals': ['Pd', 'Pt', 'Ni', 'Rh', 'Ir'],
                'pattern': 'M + R-X -> R-M-X'
            },
            'reductive_elimination': {
                'metals': ['Pd', 'Pt', 'Ni', 'Rh', 'Ir'],
                'pattern': 'R-M-R\' -> M + R-R\''
            },
            'transmetallation': {
                'metals': ['Pd', 'Ni', 'Cu', 'Au'],
                'pattern': 'R-M1 + M2-X -> R-M2 + M1-X'
            },
            'ligand_substitution': {
                'metals': ['all'],
                'pattern': 'M-L1 + L2 -> M-L2 + L1'
            },
            'insertion': {
                'metals': ['Pd', 'Ni', 'Rh', 'Co'],
                'pattern': 'M-R + X=Y -> M-X-Y-R'
            },
            'beta_hydride_elimination': {
                'metals': ['Pd', 'Ni', 'Pt'],
                'pattern': 'M-CH2-CH2-R -> M-H + CH2=CH-R'
            }
        }
    
    def identify_metal_centers(self, smiles: str) -> List[str]:
        """Identify metal atoms in SMILES"""
        metals = []
        for metal_list in self.metal_patterns.values():
            for metal in metal_list:
                if f'[{metal}' in smiles:
                    metals.append(metal)
        return metals
    
    def predict_products(self, reactants_smiles: List[str], reaction_type: str = 'auto') -> List[str]:
        """Predict products for organometallic reactions"""
        # Identify metals in reactants
        all_metals = []
        metal_compounds = []
        organic_reactants = []
        
        for smiles in reactants_smiles:
            metals = self.identify_metal_centers(smiles)
            if metals:
                all_metals.extend(metals)
                metal_compounds.append(smiles)
            else:
                organic_reactants.append(smiles)
        
        if not all_metals:
            return reactants_smiles  # No metals, return as-is
        
        # Special handling for phenolic ligands with nickel
        if 'Ni' in all_metals and organic_reactants:
            # Check if we have phenolic compounds (syringol, guaiacol, etc.)
            for org_smiles in organic_reactants:
                mol = Chem.MolFromSmiles(org_smiles)
                if mol:
                    # Check for phenolic OH
                    phenol_pattern = Chem.MolFromSmarts('c1ccccc1[OH]')
                    if mol.HasSubstructMatch(phenol_pattern):
                        # Form nickel-phenolate complex
                        # For syringol-type compounds, we get chelation
                        if 'COC' in org_smiles and org_smiles.count('OC') >= 2:
                            # Bis-syringolate nickel complex (simplified representation)
                            # In reality this would be [Ni(syringolate)2]
                            return [f"[Ni+2].[O-]c1c(OC)cccc1OC.[O-]c1c(OC)cccc1OC"]
                        else:
                            # Simple phenolate complex
                            phenolate = org_smiles.replace('O)', '[O-])')
                            return [f"[Ni+2].{phenolate}.{phenolate}"]
        
        # Auto-detect reaction type if not specified
        if reaction_type == 'auto':
            reaction_type = self.detect_reaction_type(reactants_smiles, all_metals)
        
        # Apply reaction pattern
        products = self.apply_reaction_pattern(reactants_smiles, reaction_type, all_metals)
        
        return products if products else reactants_smiles
    
    def detect_reaction_type(self, reactants_smiles: List[str], metals: List[str]) -> str:
        """Auto-detect the most likely reaction type"""
        # Simple heuristics for reaction type detection
        has_halide = any('Cl' in s or 'Br' in s or 'I' in s for s in reactants_smiles)
        has_alkene = any('C=C' in s for s in reactants_smiles)
        has_co = any('C#O' in s or '[C-]#[O+]' in s for s in reactants_smiles)
        
        if has_halide and metals[0] in ['Pd', 'Ni']:
            return 'oxidative_addition'
        elif has_co:
            return 'insertion'
        elif has_alkene:
            return 'insertion'
        else:
            return 'ligand_substitution'
    
    def apply_reaction_pattern(self, reactants_smiles: List[str], reaction_type: str, metals: List[str]) -> List[str]:
        """Apply reaction pattern to generate products"""
        
        # Special case for nickel-syringol type reactions
        if 'Ni' in metals:
            # Check for phenolic compounds
            phenolic_compounds = []
            for smiles in reactants_smiles:
                if '[Ni]' not in smiles:  # Not the metal itself
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        # Check for phenolic OH
                        if mol.HasSubstructMatch(Chem.MolFromSmarts('c[OH]')):
                            phenolic_compounds.append(smiles)
            
            if phenolic_compounds:
                # For syringol (2,6-dimethoxyphenol) and similar compounds
                # The product is typically a bis-phenolate nickel complex
                # Simplified representation: Ni(L)2 where L is the deprotonated ligand
                
                ligand = phenolic_compounds[0]
                # Convert phenol to phenolate (deprotonate the OH)
                # This is a simplified transformation
                if 'C(=CC=C' in ligand and ')O' in ligand:
                    # Try to deprotonate the phenolic OH
                    phenolate = ligand.replace(')O', ')[O-]')
                elif 'c1O' in ligand:
                    phenolate = ligand.replace('c1O', 'c1[O-]')
                else:
                    phenolate = ligand.replace('OH', '[O-]').replace('O)', '[O-])')
                
                # Form bis-phenolate complex
                # In reality, this would be square planar or octahedral
                return [f"[Ni+2].{phenolate}.{phenolate}"]
        
        # Default patterns for other reaction types
        if reaction_type == 'ligand_substitution':
            # Simple coordination complex formation
            if len(reactants_smiles) >= 2:
                return ['.'.join(reactants_smiles)]  # Form complex
        
        elif reaction_type == 'oxidative_addition':
            # Simplified oxidative addition
            products = []
            for smiles in reactants_smiles:
                if any(hal in smiles for hal in ['Br', 'Cl', 'I']):
                    # Create metal-carbon bond (simplified)
                    # This is a very simplified representation
                    metal_inserted = smiles
                    for metal in metals:
                        if 'Br' in smiles:
                            metal_inserted = smiles.replace('Br', f'][{metal}]Br')
                        elif 'Cl' in smiles:
                            metal_inserted = smiles.replace('Cl', f'][{metal}]Cl')
                        elif 'I' in smiles:
                            metal_inserted = smiles.replace('I', f'][{metal}]I')
                        break
                    products.append(metal_inserted)
            return products if products else reactants_smiles
        
        elif reaction_type == 'insertion':
            # Simplified insertion reaction
            return ['.'.join(reactants_smiles)]
        
        elif reaction_type == 'cross_coupling':
            # For cross-coupling, we need at least an organic halide and an organometallic reagent
            # This is simplified - real cross-coupling is much more complex
            organic_halides = []
            other_organics = []
            
            for smiles in reactants_smiles:
                if '[' not in smiles:  # Organic molecule
                    if any(hal in smiles for hal in ['Br', 'Cl', 'I']):
                        organic_halides.append(smiles)
                    else:
                        other_organics.append(smiles)
            
            if organic_halides and other_organics:
                # Simplified: replace halide with the other organic group
                # This is very simplified - real mechanism involves oxidative addition,
                # transmetallation, and reductive elimination
                product = organic_halides[0]
                for hal in ['Br', 'Cl', 'I']:
                    if hal in product:
                        # This is oversimplified - just for demonstration
                        product = product.replace(hal, '')
                        break
                return [product]
        
        return reactants_smiles


class ChemicalVisualizer:
    """Enhanced chemical structure visualization with multiple rendering options"""
    
    def __init__(self):
        self.available_methods = {
            'rdkit': True,
            'py3dmol': PY3DMOL_AVAILABLE,
            'plotly': PLOTLY_AVAILABLE,
            'enhanced': PIL_AVAILABLE
        }
    
    def create_enhanced_2d_structure(self, smiles, title="Molecule", size=(400, 400)):
        """Create enhanced 2D structure with properties"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return None, None
            
            # Generate 2D coordinates
            AllChem.Compute2DCoords(mol)
            
            # Calculate molecular properties
            mw = Descriptors.MolWt(mol)
            formula = rdMolDescriptors.CalcMolFormula(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            rotatable = Lipinski.NumRotatableBonds(mol)
            
            # Create the molecule image
            img = Draw.MolToImage(mol, size=size, kekulize=True)
            
            if PIL_AVAILABLE:
                # Add property information to the image
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
                
                # Add text with molecular properties
                properties_text = [
                    f"Formula: {formula}",
                    f"MW: {mw:.2f} g/mol",
                    f"LogP: {logp:.2f}",
                    f"TPSA: {tpsa:.2f} Å²",
                    f"HBD/HBA: {hbd}/{hba}",
                    f"Rotatable bonds: {rotatable}"
                ]
                
                y_offset = 10
                for prop in properties_text:
                    draw.text((10, y_offset), prop, fill="black", font=font)
                    y_offset += 20
            
            return img, {
                "formula": formula, 
                "mw": mw, 
                "logp": logp, 
                "tpsa": tpsa,
                "hbd": hbd,
                "hba": hba,
                "rotatable": rotatable
            }
        
        except Exception as e:
            print(f"Error creating enhanced 2D structure: {e}")
            return None, None
    
    def visualize_reaction_mechanism(self, reactants_smiles, products_smiles, mechanism_steps=None):
        """Create detailed reaction mechanism visualization"""
        if not PIL_AVAILABLE:
            print("PIL required for mechanism visualization")
            return None
        
        try:
            # Create a larger canvas for mechanism
            canvas_width = 1400
            canvas_height = 700
            canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
            draw = ImageDraw.Draw(canvas)
            
            # Add title
            try:
                font_large = ImageFont.truetype("arial.ttf", 20)
                font_medium = ImageFont.truetype("arial.ttf", 14)
                font_small = ImageFont.truetype("arial.ttf", 12)
            except:
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            draw.text((canvas_width//2 - 100, 20), "Reaction Mechanism", fill="black", font=font_large)
            
            # Process and visualize reactants
            x_offset = 50
            y_offset = 100
            reactant_images = []
            
            # Identify if this is a metal-carboxylate reaction
            is_metal_carboxylate = False
            metal_type = None
            
            for smiles in reactants_smiles:
                if '[Al]' in smiles:
                    metal_type = 'Al'
                    is_metal_carboxylate = any('C(=O)O' in s for s in reactants_smiles)
                elif '[Cu]' in smiles:
                    metal_type = 'Cu'
                    is_metal_carboxylate = any('C(=O)O' in s for s in reactants_smiles)
            
            # Draw reactants
            for i, smiles in enumerate(reactants_smiles):
                # Special handling for metal atoms
                if smiles in ['[Al]', '[Pd]', '[Pt]', '[Cu]', '[Fe]', '[Ni]']:
                    # Create a simple representation for bare metals
                    img = Image.new('RGB', (200, 200), 'white')
                    img_draw = ImageDraw.Draw(img)
                    metal_symbol = smiles.strip('[]')
                    
                    # Draw circle for metal
                    img_draw.ellipse([70, 70, 130, 130], outline="black", width=2)
                    img_draw.text((85, 85), metal_symbol, fill="black", font=font_large)
                    
                    canvas.paste(img, (x_offset, y_offset))
                    
                    # Add label
                    draw.text((x_offset + 50, y_offset + 210), f"{metal_symbol} metal", fill="gray", font=font_small)
                else:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        AllChem.Compute2DCoords(mol)
                        img = Draw.MolToImage(mol, size=(200, 200))
                        canvas.paste(img, (x_offset, y_offset))
                        
                        # Add label for special compounds
                        if 'CCCCCCCCCCCCCCCCCC(=O)O' in smiles:
                            draw.text((x_offset + 30, y_offset + 210), "Stearic Acid", fill="gray", font=font_small)
                        elif 'COc1cc(C)cc(OC)c1O' in smiles or 'COC1=C(C(=CC=C1)OC)O' in smiles:
                            draw.text((x_offset + 40, y_offset + 210), "Syringol", fill="gray", font=font_small)
                
                x_offset += 220
                if i < len(reactants_smiles) - 1:
                    draw.text((x_offset - 20, y_offset + 90), "+", fill="black", font=font_large)
            
            # Add arrow with reaction conditions
            arrow_x = x_offset + 20
            arrow_length = 120
            
            # Draw arrow
            draw.line([(arrow_x, y_offset + 100), (arrow_x + arrow_length - 20, y_offset + 100)], fill="black", width=3)
            draw.polygon([(arrow_x + arrow_length - 20, y_offset + 90), 
                         (arrow_x + arrow_length, y_offset + 100), 
                         (arrow_x + arrow_length - 20, y_offset + 110)], fill="black")
            
            # Add reaction condition label
            if is_metal_carboxylate:
                draw.text((arrow_x + 20, y_offset + 60), "Salt formation", fill="blue", font=font_medium)
                draw.text((arrow_x + 30, y_offset + 120), "-H₂↑", fill="red", font=font_small)
            elif any('[' in s and ']' in s for s in reactants_smiles):
                draw.text((arrow_x + 20, y_offset + 60), "coordination", fill="blue", font=font_medium)
            
            # Add products
            x_offset = arrow_x + arrow_length + 20
            
            for i, smiles in enumerate(products_smiles):
                # Handle complex products with multiple components
                if '[Al+3]' in smiles and ('C(=O)[O-]' in smiles or 'c(=O)[O-]' in smiles):
                    # This is aluminum carboxylate - draw special representation
                    img = Image.new('RGB', (250, 250), 'white')
                    img_draw = ImageDraw.Draw(img)
                    
                    # Draw central Al3+
                    img_draw.ellipse([110, 110, 140, 140], outline="blue", width=2, fill="lightblue")
                    img_draw.text((115, 115), "Al³⁺", fill="blue", font=font_medium)
                    
                    # Draw three carboxylate groups around it
                    positions = [(125, 70), (80, 140), (170, 140)]
                    for px, py in positions:
                        img_draw.text((px, py), "RCOO⁻", fill="red", font=font_small)
                        # Draw line from carboxylate to Al
                        img_draw.line([(px + 20, py + 10), (125, 125)], fill="gray", width=1)
                    
                    # Add label
                    img_draw.text((50, 220), "Aluminum Stearate", fill="black", font=font_small)
                    
                    canvas.paste(img, (x_offset, y_offset - 25))
                    x_offset += 270
                    
                elif '[Cu+2]' in smiles and 'C(=O)[O-]' in smiles:
                    # Copper carboxylate
                    img = Image.new('RGB', (250, 250), 'white')
                    img_draw = ImageDraw.Draw(img)
                    
                    # Draw central Cu2+
                    img_draw.ellipse([110, 110, 140, 140], outline="green", width=2, fill="lightgreen")
                    img_draw.text((115, 115), "Cu²⁺", fill="green", font=font_medium)
                    
                    # Draw two carboxylate groups
                    positions = [(80, 115), (170, 115)]
                    for px, py in positions:
                        img_draw.text((px, py), "RCOO⁻", fill="red", font=font_small)
                        img_draw.line([(px + 20, py + 5), (125, 125)], fill="gray", width=1)
                    
                    img_draw.text((50, 220), "Copper Carboxylate", fill="black", font=font_small)
                    
                    canvas.paste(img, (x_offset, y_offset - 25))
                    x_offset += 270
                    
                elif '[H][H]' in smiles or smiles == '[H][H]':
                    # Hydrogen gas
                    img = Image.new('RGB', (150, 200), 'white')
                    img_draw = ImageDraw.Draw(img)
                    img_draw.text((50, 90), "H₂↑", fill="red", font=font_large)
                    img_draw.text((40, 120), "(gas)", fill="gray", font=font_small)
                    canvas.paste(img, (x_offset, y_offset))
                    x_offset += 170
                    
                elif '[Al+3].[OH-].[OH-].[OH-]' in smiles:
                    # Aluminum hydroxide
                    img = Image.new('RGB', (200, 200), 'white')
                    img_draw = ImageDraw.Draw(img)
                    img_draw.text((50, 90), "Al(OH)₃", fill="blue", font=font_large)
                    canvas.paste(img, (x_offset, y_offset))
                    x_offset += 220
                    
                elif '.' in smiles:
                    # Other complex products
                    parts = smiles.split('.')
                    for part in parts:
                        if part.strip():
                            try:
                                mol = Chem.MolFromSmiles(part)
                                if mol:
                                    AllChem.Compute2DCoords(mol)
                                    img = Draw.MolToImage(mol, size=(150, 150))
                                    canvas.paste(img, (x_offset, y_offset + 25))
                                    x_offset += 160
                            except:
                                pass
                else:
                    # Regular molecules
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        AllChem.Compute2DCoords(mol)
                        img = Draw.MolToImage(mol, size=(200, 200))
                        canvas.paste(img, (x_offset, y_offset))
                        x_offset += 220
                
                if i < len(products_smiles) - 1:
                    draw.text((x_offset - 20, y_offset + 90), "+", fill="black", font=font_large)
            
            # Add mechanism description at bottom
            y_text = 400
            if mechanism_steps:
                for step in mechanism_steps:
                    draw.text((50, y_text), step, fill="black", font=font_small)
                    y_text += 25
            else:
                # Add default mechanism description based on reaction type
                if is_metal_carboxylate and metal_type:
                    steps = [
                        f"Mechanism: {metal_type} metal reacts with carboxylic acid",
                        f"1. Carboxylic acid proton is displaced by {metal_type}³⁺/²⁺",
                        f"2. Three (Al) or two (Cu) carboxylate anions coordinate to the metal center",
                        f"3. Hydrogen gas (H₂) is evolved as a byproduct",
                        f"Product: Metal carboxylate salt (used in soaps, lubricants, catalysts)"
                    ]
                    for step in steps:
                        draw.text((50, y_text), step, fill="gray", font=font_small)
                        y_text += 20
                elif any('[' in s and ']' in s for s in reactants_smiles):
                    draw.text((50, y_text), "Mechanism: Metal coordination with organic ligand", fill="gray", font=font_small)
            
            # Add reaction stoichiometry for metal carboxylate reactions
            if is_metal_carboxylate:
                if metal_type == 'Al':
                    draw.text((50, 550), "Stoichiometry: 3 RCOOH + Al → Al(RCOO)₃ + 3/2 H₂", fill="black", font=font_medium)
                elif metal_type == 'Cu':
                    draw.text((50, 550), "Stoichiometry: 2 RCOOH + Cu → Cu(RCOO)₂ + H₂", fill="black", font=font_medium)
            
            filename = "reaction_mechanism.png"
            canvas.save(filename)
            return filename
            
        except Exception as e:
            print(f"Error creating mechanism visualization: {e}")
            return None


class ReactionPredictor:
    def __init__(self):
        # Initialize components
        self.visualizer = ChemicalVisualizer()
        self.db_interface = ChemicalDatabaseInterface()
        self.organometallic_predictor = OrganometallicReactionPredictor()
        
        # Available models configuration
        self.available_models = {
            "rxnfp": {
                "name": "rxnfp/transformers",
                "description": "RXNFP - Reaction Fingerprints model",
                "loaded": False,
                "tokenizer": None,
                "model": None
            },
            "molecular_transformer": {
                "name": "Molecular-Transformer",
                "description": "Molecular Transformer for reaction prediction",
                "loaded": False,
                "tokenizer": None,
                "model": None
            }
        }
        
        self.selected_model = None
        
        # Constants for reaction rate calculation
        self.R = 8.314  # Gas constant (J/mol·K)
        
        # Dictionary mapping reaction conditions to typical parameters
        self.condition_params = {
            # Standard organic reactions
            "sn2": {"A": 1e10, "Ea": 60000, "order": 2},
            "sn1": {"A": 1e13, "Ea": 80000, "order": 1},
            "e2": {"A": 1e11, "Ea": 70000, "order": 2},
            "e1": {"A": 1e12, "Ea": 85000, "order": 1},
            "addition": {"A": 1e10, "Ea": 50000, "order": 2},
            "elimination": {"A": 1e11, "Ea": 75000, "order": 1},
            "substitution": {"A": 1e10, "Ea": 65000, "order": 2},
            "radical": {"A": 1e14, "Ea": 90000, "order": 1},
            
            # Organometallic reactions
            "oxidative_addition": {"A": 1e9, "Ea": 40000, "order": 2},
            "reductive_elimination": {"A": 1e10, "Ea": 55000, "order": 1},
            "transmetallation": {"A": 1e10, "Ea": 45000, "order": 2},
            "ligand_exchange": {"A": 1e8, "Ea": 35000, "order": 1},
            "insertion": {"A": 1e10, "Ea": 50000, "order": 2},
            "metathesis": {"A": 1e10, "Ea": 70000, "order": 2},
            
            # Catalytic processes
            "hydrogenation": {"A": 1e11, "Ea": 60000, "order": 1},
            "oxidation": {"A": 1e13, "Ea": 100000, "order": 1},
            "cross_coupling": {"A": 1e11, "Ea": 75000, "order": 2},
            "carbonylation": {"A": 1e12, "Ea": 85000, "order": 1},
            
            # Special conditions requested
            "pyrolysis": {"A": 1e15, "Ea": 180000, "order": 1},  # High temperature decomposition
            "combustion": {"A": 1e14, "Ea": 120000, "order": 2},  # Oxidation with O2
            "electrochemical": {"A": 1e8, "Ea": 45000, "order": 1},  # More realistic for metal-acid reactions
            
            # Metal-specific reactions (adjusted for realism)
            "metal_reduction": {"A": 1e9, "Ea": 65000, "order": 1},  # Al, Cu reduction - slower
            "metal_oxidation": {"A": 1e10, "Ea": 75000, "order": 1},  # Al2O3, CuO oxidation
            "metal_coordination": {"A": 1e8, "Ea": 40000, "order": 1},  # Metal complex formation
            "metal_carboxylate": {"A": 1e7, "Ea": 55000, "order": 1},  # Metal + fatty acid reactions
            
            # Solvent/polymer interactions
            "hydrolysis": {"A": 1e10, "Ea": 50000, "order": 1},  # Water reactions
            "acetylation": {"A": 1e11, "Ea": 60000, "order": 2},  # Acetic acid reactions
            "polymer_encapsulation": {"A": 1e8, "Ea": 25000, "order": 1},  # PVAc interactions
            
            # Biomass-specific reactions
            "biomass_pyrolysis": {"A": 1e14, "Ea": 150000, "order": 1},
            "lignin_degradation": {"A": 1e13, "Ea": 140000, "order": 1},
            "cellulose_hydrolysis": {"A": 1e11, "Ea": 110000, "order": 1},
            
            # General conditions
            "ideal": {"A": 1e12, "Ea": 80000, "order": 1},
        }
        
        # Compound database for special molecules in your list
        self.special_compounds = {
            # Tree of Heaven compounds
            "canthine-6-one": "Canthine-6-one",
            "1-methoxy-canthine-6-one": "1-Methoxy-canthine-6-one",
            "canthine-6-one-3-n-oxide": "Canthine-6-one-3-N-oxide",
            "ailanthone": "Ailanthone",
            "chapparin": "Chapparin",
            "shinjulactone": "Shinjulactone",
            "shinjulactone b": "Shinjulactone B",
            "dehydroglaucarubinone": "Δ13(18)-Dehydroglaucarubinone",
            "dehydroglaucarubolone": "Δ13(18)-Dehydroglaucarubolone",
            "ailantinol a": "Ailantinol A",
            "ailantinol b": "Ailantinol B",
            "ailantinol c": "Ailantinol C",
            "ailantinol d": "Ailantinol D",
            "ailantinol e": "Ailantinol E",
            "ailantinol f": "Ailantinol F",
            "ailantinol g": "Ailantinol G",
            "ailantinol h": "Ailantinol H",
            "altissimacoumarin a": "Altissimacoumarin A",
            "altissimacoumarin b": "Altissimacoumarin B",
            
            # Common biomass compounds
            "syringol": "2,6-dimethoxyphenol",
            "guaiacol": "2-methoxyphenol",
            "levoglucosan": "Levoglucosan",
            "hmf": "5-Hydroxymethylfurfural",
            "5-hydroxymethylfurfural": "5-Hydroxymethylfurfural",
            
            # Fatty acids
            "palmitic acid": "n-Hexadecanoic Acid",
            "linoleic acid": "9,12-Octadecadienoic Acid",
            "stearic acid": "Octadecanoic Acid",
            "erucic acid": "Erucic Acid",
            "eicosenoic acid": "cis-13-Eicosenoic Acid",
            
            # Sterols
            "stigmasterol": "Stigmasterol",
            "campesterol": "Campesterol",
            "γ-tocopherol": "γ-Tocopherol",
            "gamma-tocopherol": "γ-Tocopherol",
            
            # Metals and metal compounds
            "aluminum": "Al",
            "aluminium": "Al",
            "aluminum oxide": "Al2O3",
            "alumina": "Al2O3",
            "copper": "Cu",
            "copper oxide": "CuO",
            "cupric oxide": "CuO",
            "nickel": "Ni",
            
            # Polymer
            "pvac": "PVAc",
            "polyvinyl acetate": "PVAc",
            
            # Generic representatives
            "isothiocyanates": "R-N=C=S",
            "nitriles": "R-C≡N",
            "phenols": "Ar-OH",
            "alkenes": "RCH=CHR",
            "alkanes": "R-CH3",
            "aldehydes": "R-CHO",
            "ketones": "R-CO-R'",
            "amino acids": "H2N-CHR-COOH",
            "alcohols": "R-CH2OH",
            "fatty acids": "R-COOH",
        }

    def get_smiles_from_name(self, compound_name: str) -> Optional[str]:
        """Get SMILES from compound name using the database interface and special handling"""
        
        # First check if it's a special compound in our database
        name_lower = compound_name.lower().strip()
        
        # Check special compounds database
        if hasattr(self, 'special_compounds') and name_lower in self.special_compounds:
            normalized_name = self.special_compounds[name_lower]
            compound_name = normalized_name
        
        # Handle specific cases for metals and metal oxides
        metal_smiles = {
            "al": "[Al]",
            "aluminum": "[Al]",
            "aluminium": "[Al]",
            "al2o3": "[Al+3].[Al+3].[O-2].[O-2].[O-2]",
            "aluminum oxide": "[Al+3].[Al+3].[O-2].[O-2].[O-2]",
            "alumina": "[Al+3].[Al+3].[O-2].[O-2].[O-2]",
            "cu": "[Cu]",
            "copper": "[Cu]",
            "cuo": "[Cu+2].[O-2]",
            "copper oxide": "[Cu+2].[O-2]",
            "cupric oxide": "[Cu+2].[O-2]",
            "ni": "[Ni]",
            "nickel": "[Ni]",
            "pvac": "CC(=O)O[C@@H]1[C@H](OC(C)=O)[C@@H](COC(C)=O)O[C@H](COC(C)=O)[C@@H]1OC(C)=O",  # Simplified PVAc unit
            "polyvinyl acetate": "CC(=O)O[C@@H]1[C@H](OC(C)=O)[C@@H](COC(C)=O)O[C@H](COC(C)=O)[C@@H]1OC(C)=O",
            "water": "O",
            "h2o": "O",
            "acetic acid": "CC(=O)O",
            "ch3cooh": "CC(=O)O",
            "methanol": "CO",
            "ch3oh": "CO",
            "carbon monoxide": "[C-]#[O+]",
            "co": "[C-]#[O+]",
            "carbon dioxide": "O=C=O",
            "co2": "O=C=O",
            "ammonia": "N",
            "nh3": "N",
            "hydrogen cyanide": "C#N",
            "hcn": "C#N",
            "hydrogen sulfide": "S",
            "h2s": "S",
        }
        
        if name_lower in metal_smiles:
            return metal_smiles[name_lower]
        
        # Handle generic representatives with specific SMILES patterns
        generic_patterns = {
            "isothiocyanate": "CN=C=S",  # Methyl isothiocyanate as example
            "r-n=c=s": "CN=C=S",
            "nitrile": "CC#N",  # Acetonitrile as example
            "r-c≡n": "CC#N",
            "phenol": "Oc1ccccc1",
            "ar-oh": "Oc1ccccc1",
            "alkene": "C=C",  # Ethylene as example
            "rch=chr": "C=C",
            "alkane": "CC",  # Ethane as example
            "aldehyde": "CC=O",  # Acetaldehyde as example
            "r-cho": "CC=O",
            "ketone": "CC(=O)C",  # Acetone as example
            "r-co-r": "CC(=O)C",
            "r-co-r'": "CC(=O)C",
            "amino acid": "NC(C)C(=O)O",  # Alanine as example
            "h2n-chr-cooh": "NC(C)C(=O)O",
            "alcohol": "CCO",  # Ethanol as example
            "r-ch2oh": "CCO",
            "fatty acid": "CCCCCCCC(=O)O",  # Octanoic acid as example
            "r-cooh": "CCCCCCCC(=O)O",
            "hydrocarbon": "CCCC",  # Butane as example
            "voc": "CC(C)C",  # Isobutane as example
            "protein": "NC(C)C(=O)NC(C)C(=O)O",  # Dipeptide as example
            "glucosinolate": "CC(=NOS(=O)(=O)O)SC1OC(CO)C(O)C(O)C1O",  # Simplified glucosinolate
            "flavonoid": "O=C1CC(c2ccccc2)Oc2ccccc21",  # Flavanone as example
            "alkaloid": "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC4C(C=C3)O",  # Morphine as example
            "furan": "c1ccoc1",
            "pyrrole": "c1cc[nH]c1",
            "indole": "c1ccc2c(c1)cc[nH]2",
            "coumarin": "O=C1Oc2ccccc2C=C1",
            "scopoletin": "COc1cc2ccc(=O)oc2cc1O",
            "isofraxidin": "COc1c2ccoc2cc2oc(=O)ccc12",
        }
        
        # Check for generic patterns
        for pattern_key, smiles in generic_patterns.items():
            if pattern_key in name_lower:
                return smiles
        
        # Use the database interface for everything else
        return self.db_interface.get_smiles(compound_name)
    
    def analyze_reaction_type(self, reactants_smiles: List[str]) -> Dict[str, Any]:
        """Analyze the reaction type based on reactants"""
        analysis = {
            'has_metal': False,
            'metals': [],
            'has_organic': False,
            'functional_groups': [],
            'reaction_class': 'unknown',
            'suggested_conditions': [],
            'auto_condition': None  # Automatically selected condition
        }
        
        for smiles in reactants_smiles:
            # Check for metals
            metals = self.organometallic_predictor.identify_metal_centers(smiles)
            if metals:
                analysis['has_metal'] = True
                analysis['metals'].extend(metals)
            
            # Check for organic compounds
            if 'C' in smiles or 'c' in smiles:
                analysis['has_organic'] = True
                
                # Identify functional groups
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # Check for common functional groups
                    if mol.HasSubstructMatch(Chem.MolFromSmarts('[CX3](=O)[OX2H1]')):
                        analysis['functional_groups'].append('carboxylic_acid')
                    if mol.HasSubstructMatch(Chem.MolFromSmarts('[CX3](=O)[OX2]')):
                        analysis['functional_groups'].append('ester')
                    if mol.HasSubstructMatch(Chem.MolFromSmarts('[CX3](=O)[NX3]')):
                        analysis['functional_groups'].append('amide')
                    if mol.HasSubstructMatch(Chem.MolFromSmarts('[OX2H]')):
                        analysis['functional_groups'].append('alcohol')
                    if mol.HasSubstructMatch(Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]')):
                        analysis['functional_groups'].append('amine')
                    if mol.HasSubstructMatch(Chem.MolFromSmarts('C=C')):
                        analysis['functional_groups'].append('alkene')
                    if mol.HasSubstructMatch(Chem.MolFromSmarts('C#C')):
                        analysis['functional_groups'].append('alkyne')
                    if mol.HasSubstructMatch(Chem.MolFromSmarts('[F,Cl,Br,I]')):
                        analysis['functional_groups'].append('halide')
        
        # Determine reaction class and auto-select condition
        if analysis['has_metal'] and analysis['has_organic']:
            analysis['reaction_class'] = 'organometallic'
            
            # Check for specific reaction types
            if 'carboxylic_acid' in analysis['functional_groups']:
                # Metal + carboxylic acid = salt formation
                analysis['auto_condition'] = 'metal_carboxylate'
                analysis['suggested_conditions'] = ['metal_carboxylate', 'electrochemical']
            elif 'Al' in analysis['metals'] or 'Cu' in analysis['metals']:
                if any(fg in analysis['functional_groups'] for fg in ['alcohol', 'ester', 'amide']):
                    analysis['auto_condition'] = 'metal_reduction'
                    analysis['suggested_conditions'] = ['metal_reduction', 'electrochemical']
                else:
                    analysis['auto_condition'] = 'metal_coordination'
                    analysis['suggested_conditions'] = ['metal_coordination', 'ligand_exchange']
            elif 'Pd' in analysis['metals']:
                analysis['auto_condition'] = 'cross_coupling'
                analysis['suggested_conditions'] = ['cross_coupling', 'oxidative_addition', 'reductive_elimination']
            elif 'Ni' in analysis['metals']:
                if 'halide' in analysis['functional_groups']:
                    analysis['auto_condition'] = 'cross_coupling'
                    analysis['suggested_conditions'] = ['cross_coupling', 'oxidative_addition']
                else:
                    analysis['auto_condition'] = 'hydrogenation'
                    analysis['suggested_conditions'] = ['hydrogenation', 'metal_coordination']
            else:
                analysis['auto_condition'] = 'metal_coordination'
                analysis['suggested_conditions'] = ['ligand_exchange', 'insertion', 'oxidative_addition']
        
        elif 'Al+3' in str(reactants_smiles) or 'Cu+2' in str(reactants_smiles):
            # Metal oxides
            analysis['reaction_class'] = 'metal_oxide'
            analysis['auto_condition'] = 'metal_oxidation'
            analysis['suggested_conditions'] = ['metal_oxidation', 'oxidation']
        
        elif 'O' in reactants_smiles:  # Water present
            analysis['reaction_class'] = 'hydrolysis'
            analysis['auto_condition'] = 'hydrolysis'
            analysis['suggested_conditions'] = ['hydrolysis']
        
        elif 'CC(=O)O' in reactants_smiles:  # Acetic acid
            analysis['reaction_class'] = 'acetylation'
            analysis['auto_condition'] = 'acetylation'
            analysis['suggested_conditions'] = ['acetylation']
        
        elif analysis['has_organic']:
            analysis['reaction_class'] = 'organic'
            
            # Suggest conditions based on functional groups
            if 'halide' in analysis['functional_groups']:
                if 'amine' in analysis['functional_groups'] or 'alcohol' in analysis['functional_groups']:
                    analysis['auto_condition'] = 'sn2'
                    analysis['suggested_conditions'] = ['sn2', 'substitution']
                else:
                    analysis['auto_condition'] = 'sn2'
                    analysis['suggested_conditions'] = ['sn1', 'sn2', 'e2']
            elif 'alkene' in analysis['functional_groups']:
                analysis['auto_condition'] = 'addition'
                analysis['suggested_conditions'] = ['addition', 'oxidation', 'hydrogenation']
            elif 'alcohol' in analysis['functional_groups']:
                analysis['auto_condition'] = 'oxidation'
                analysis['suggested_conditions'] = ['elimination', 'oxidation', 'substitution']
            else:
                analysis['auto_condition'] = 'ideal'
                analysis['suggested_conditions'] = ['ideal', 'substitution']
        
        else:
            analysis['reaction_class'] = 'inorganic'
            analysis['auto_condition'] = 'ideal'
            analysis['suggested_conditions'] = ['ideal', 'electrochemical']
        
        return analysis
    
    def predict_products(self, reactants_smiles: List[str], reaction_analysis: Dict[str, Any]) -> List[str]:
        """Predict reaction products based on reactants and analysis"""
        
        # Special handling for specific metal reactions from your list
        metal_reactants = []
        metal_oxide_reactants = []
        organic_reactants = []
        
        # Categorize reactants
        for smiles in reactants_smiles:
            if '[Al]' in smiles:
                metal_reactants.append('Al')
            elif '[Cu]' in smiles:
                metal_reactants.append('Cu')
            elif '[Ni]' in smiles:
                metal_reactants.append('Ni')
            elif '[Fe]' in smiles:
                metal_reactants.append('Fe')
            elif 'Al+3' in smiles and 'O-2' in smiles:
                metal_oxide_reactants.append('Al2O3')
            elif 'Cu+2' in smiles and 'O-2' in smiles:
                metal_oxide_reactants.append('CuO')
            else:
                organic_reactants.append(smiles)
        
        # Handle specific reaction types from your Excel file
        products = []
        
        # SAFEGUARD: Check for carboxylic acids FIRST before other functional groups
        has_carboxylic_acid = False
        carboxylic_acids = []
        other_organics = []
        
        for org in organic_reactants:
            # More robust carboxylic acid detection
            if any(pattern in org for pattern in ['C(=O)O', 'c(=O)O', 'C(O)=O', 'COOH', 'C(=O)[OH]']):
                has_carboxylic_acid = True
                carboxylic_acids.append(org)
            else:
                other_organics.append(org)
        
        # Metal + Carboxylic Acid reactions (PRIORITY)
        if metal_reactants and has_carboxylic_acid:
            for metal in metal_reactants:
                for acid in carboxylic_acids:
                    # Form metal carboxylate salt
                    # Remove the acidic hydrogen to form carboxylate anion
                    carboxylate = acid
                    for pattern, replacement in [
                        ('C(=O)O', 'C(=O)[O-]'),
                        ('c(=O)O', 'c(=O)[O-]'),
                        ('C(O)=O', 'C([O-])=O'),
                        ('COOH', 'COO-'),
                        ('C(=O)[OH]', 'C(=O)[O-]'),
                        ('C(=O)OH', 'C(=O)[O-]')
                    ]:
                        if pattern in carboxylate:
                            carboxylate = carboxylate.replace(pattern, replacement)
                            break
                    
                    # Form appropriate metal salt based on metal valency
                    if metal == 'Al':
                        # Aluminum forms trivalent salts
                        products.append(f"[Al+3].{carboxylate}.{carboxylate}.{carboxylate}")
                    elif metal in ['Cu', 'Ni', 'Fe']:
                        # Divalent metal salts
                        products.append(f"[{metal}+2].{carboxylate}.{carboxylate}")
                    else:
                        # Generic metal salt
                        products.append(f"[{metal}+].{carboxylate}")
                    
                    # Always produce H2 gas with metal-acid reactions
                    if '[H][H]' not in products:
                        products.append('[H][H]')
            
            # Add any other non-acid organics unchanged
            products.extend(other_organics)
        
        # Aluminum reactions (non-carboxylic acid)
        elif 'Al' in metal_reactants and other_organics:
            for org in other_organics:
                # SAFEGUARD: Double-check this isn't a carboxylic acid
                if any(p in org for p in ['C(=O)O', 'COOH', 'C(O)=O']):
                    # This is a carboxylic acid that wasn't caught - form salt
                    carboxylate = org.replace('C(=O)O', 'C(=O)[O-]').replace('COOH', 'COO-')
                    products.append(f"[Al+3].{carboxylate}.{carboxylate}.{carboxylate}")
                    products.append('[H][H]')
                elif 'C=O' in org or 'c=o' in org:  # Other carbonyls (aldehydes, ketones)
                    # Reduce carbonyl to alcohol
                    product = org.replace('C=O', 'C(O)').replace('c=o', 'c(O)')
                    products.append(product)
                    products.append('[Al+3].[OH-].[OH-].[OH-]')  # Al(OH)3
                elif 'C=C' in org:  # Alkene reduction
                    product = org.replace('C=C', 'CC')
                    products.append(product)
                    products.append('[Al+3].[OH-].[OH-].[OH-]')
                elif 'OH' in org or 'O)' in org:  # Phenolic or alcoholic compounds
                    # Form alkoxide complex
                    alkoxide = org.replace('OH', '[O-]').replace('O)', '[O-])')
                    products.append(f"[Al+3].{alkoxide}.{alkoxide}.{alkoxide}")
                else:
                    # Generic coordination
                    products.append(f"[Al+3].{org}.{org}.{org}")
        
        # Aluminum oxide oxidation reactions
        elif 'Al2O3' in metal_oxide_reactants and organic_reactants:
            for org in organic_reactants:
                # SAFEGUARD: Check reaction type
                if 'OH' in org and 'C' in org:  # Alcohols
                    # Oxidize alcohol to aldehyde/ketone
                    if 'CH2OH' in org or 'CH(OH)' in org:
                        oxidized = org.replace('CH2OH', 'CHO').replace('CH(OH)', 'C(=O)')
                        products.append(oxidized)
                    else:
                        products.append(org)  # No change if can't oxidize
                elif 'C=C' in org:  # Alkene oxidation
                    # Form epoxide or diol (simplified)
                    products.append(org.replace('C=C', 'C(O)C(O)'))
                else:
                    products.append(org)  # No reaction
                products.append('O')  # Water byproduct
        
        # Copper reactions (non-carboxylic acid)
        elif 'Cu' in metal_reactants and other_organics:
            for org in other_organics:
                # SAFEGUARD: Double-check for carboxylic acid
                if any(p in org for p in ['C(=O)O', 'COOH', 'C(O)=O']):
                    carboxylate = org.replace('C(=O)O', 'C(=O)[O-]').replace('COOH', 'COO-')
                    products.append(f"[Cu+2].{carboxylate}.{carboxylate}")
                    products.append('[H][H]')
                elif 'C(=O)' in org or 'c(=O)' in org:  # Carbonyl reduction
                    product = org.replace('C(=O)', 'C(O)').replace('c(=O)', 'c(O)')
                    products.append(product)
                    products.append('[Cu+2].[OH-].[OH-]')  # Cu(OH)2
                else:
                    products.append(f"[Cu+2].{org}.{org}")
        
        # Copper oxide oxidation reactions
        elif 'CuO' in metal_oxide_reactants and organic_reactants:
            for org in organic_reactants:
                if 'OH' in org:
                    # Oxidize alcohols
                    oxidized = org.replace('CH2OH', 'CHO').replace('CH(OH)', 'C(=O)')
                    products.append(oxidized)
                else:
                    products.append(org)  # No change
                products.append('O')  # Water
        
        # Water hydrolysis reactions
        elif 'O' in reactants_smiles and len(reactants_smiles) > 1:  # Water with something else
            for smiles in reactants_smiles:
                if smiles != 'O':
                    if 'C#N' in smiles:  # Nitrile hydrolysis
                        # Convert nitrile to carboxylic acid
                        products.append(smiles.replace('C#N', 'C(=O)O'))
                    elif 'N=C=S' in smiles:  # Isothiocyanate hydrolysis
                        products.append('CN')  # Amine
                        products.append('O=C=S')  # COS
                    elif 'C(=O)OC' in smiles:  # Ester hydrolysis
                        # Simplified ester hydrolysis
                        products.append(smiles.replace('C(=O)OC', 'C(=O)O'))
                        products.append('CO')  # Methanol
                    else:
                        products.append(f"{smiles}.O")  # Hydrated form
        
        # Acetic acid reactions (acetylation)
        elif 'CC(=O)O' in reactants_smiles:
            for smiles in reactants_smiles:
                if smiles != 'CC(=O)O':
                    if 'OH' in smiles:  # Any hydroxyl group
                        # Acetylation
                        products.append(smiles.replace('OH', 'OC(=O)C'))
                    elif 'NH2' in smiles:  # Primary amine
                        products.append(smiles.replace('NH2', 'NHC(=O)C'))
                    elif 'NH' in smiles and not 'N=' in smiles:  # Secondary amine
                        products.append(smiles.replace('NH', 'N(C(=O)C)'))
                    else:
                        products.append(smiles)  # No reaction
                    products.append('O')  # Water byproduct
        
        # PVAc encapsulation
        elif any('OC(C)=O' in s for s in reactants_smiles):  # PVAc
            pvac = next(s for s in reactants_smiles if 'OC(C)=O' in s)
            other = next((s for s in reactants_smiles if s != pvac), None)
            if other:
                products.append(f"[{other}@PVAc]")  # Encapsulated form
        
        # SAFEGUARD: Final check for any missed carboxylic acids
        if not products and metal_reactants:
            for org in organic_reactants:
                # Last chance to catch carboxylic acids
                if 'O' in org and 'C' in org:
                    mol = Chem.MolFromSmiles(org)
                    if mol and mol.HasSubstructMatch(Chem.MolFromSmarts('[CX3](=O)[OX2H1]')):
                        # This is definitely a carboxylic acid
                        carboxylate = org.replace('OH', '[O-]').replace('O)', '[O-])')
                        if 'Al' in metal_reactants:
                            products.append(f"[Al+3].{carboxylate}.{carboxylate}.{carboxylate}")
                        elif 'Cu' in metal_reactants:
                            products.append(f"[Cu+2].{carboxylate}.{carboxylate}")
                        products.append('[H][H]')
                        break
        
        # For organometallic reactions, use specialized predictor
        if not products and reaction_analysis['reaction_class'] == 'organometallic':
            products = self.organometallic_predictor.predict_products(reactants_smiles)
            if products and products != reactants_smiles:
                return products
        
        # Try AI model if available and no specific pattern matched
        if not products and self.selected_model and self.available_models[self.selected_model]["loaded"]:
            try:
                model_products = self.query_model_for_prediction(reactants_smiles)
                if model_products:
                    return model_products
            except:
                pass
        
        # Return products if found, otherwise use heuristics
        if products:
            return products
        
        # Fallback: heuristic predictions based on reaction class
        return self.heuristic_product_prediction(reactants_smiles, reaction_analysis)
    
    def heuristic_product_prediction(self, reactants_smiles: List[str], analysis: Dict[str, Any]) -> List[str]:
        """Fallback heuristic predictions"""
        products = []
        
        if analysis['reaction_class'] == 'organic':
            # Simple heuristics for common organic reactions
            for smiles in reactants_smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # Example: Simple oxidation of alcohol to aldehyde/ketone
                    if 'alcohol' in analysis['functional_groups']:
                        # This is a simplified example
                        products.append(smiles)  # Would need actual transformation
                    else:
                        products.append(smiles)
        
        elif analysis['reaction_class'] == 'organometallic':
            # Use organometallic predictor
            products = self.organometallic_predictor.predict_products(reactants_smiles)
        
        else:
            # Return reactants as-is for unknown reactions
            products = reactants_smiles
        
        return products if products else reactants_smiles
    
    def query_model_for_prediction(self, reactants_smiles: List[str]) -> List[str]:
        """Query AI model for predictions (placeholder for actual implementation)"""
        # This would interface with actual AI models
        # For now, returning empty to trigger fallback
        return []
    
    def calculate_stoichiometry(self, reactants_smiles: List[str], products_smiles: List[str], 
                               metal_reactants: List[str] = None) -> Dict[str, Any]:
        """Calculate stoichiometric coefficients for the reaction"""
        stoich = {
            'reactant_coeffs': {},
            'product_coeffs': {},
            'balanced_equation': '',
            'reaction_type': ''
        }
        
        # Identify reaction type for stoichiometry
        if metal_reactants:
            if any('Al' in m for m in metal_reactants):
                # Aluminum reactions
                for r_smiles in reactants_smiles:
                    if any(p in r_smiles for p in ['C(=O)O', 'COOH']):
                        # Al + carboxylic acid
                        stoich['reaction_type'] = 'metal_carboxylate_Al'
                        stoich['reactant_coeffs'] = {'Al': 1, 'RCOOH': 3}
                        stoich['product_coeffs'] = {'Al(RCOO)3': 1, 'H2': 1.5}
                        break
                else:
                    stoich['reaction_type'] = 'metal_reduction'
                    stoich['reactant_coeffs'] = {'Al': 2, 'Organic': 3}
                    stoich['product_coeffs'] = {'Product': 3, 'Al(OH)3': 2}
                    
            elif any('Cu' in m for m in metal_reactants):
                # Copper reactions
                for r_smiles in reactants_smiles:
                    if any(p in r_smiles for p in ['C(=O)O', 'COOH']):
                        stoich['reaction_type'] = 'metal_carboxylate_Cu'
                        stoich['reactant_coeffs'] = {'Cu': 1, 'RCOOH': 2}
                        stoich['product_coeffs'] = {'Cu(RCOO)2': 1, 'H2': 1}
                        break
                else:
                    stoich['reaction_type'] = 'metal_reduction'
                    stoich['reactant_coeffs'] = {'Cu': 1, 'Organic': 1}
                    stoich['product_coeffs'] = {'Product': 1, 'Cu(OH)2': 1}
                    
            elif any('Ni' in m for m in metal_reactants):
                # Nickel reactions
                stoich['reaction_type'] = 'metal_coordination'
                stoich['reactant_coeffs'] = {'Ni': 1, 'Ligand': 2}
                stoich['product_coeffs'] = {'Ni(L)2': 1}
        
        # Metal oxide reactions
        elif any('Al2O3' in str(s) or 'Al+3' in str(s) and 'O-2' in str(s) for s in reactants_smiles):
            stoich['reaction_type'] = 'metal_oxide_oxidation'
            stoich['reactant_coeffs'] = {'Al2O3': 1, 'Organic': 2}
            stoich['product_coeffs'] = {'Oxidized': 2, 'H2O': 1}
            
        elif any('CuO' in str(s) or 'Cu+2' in str(s) and 'O-2' in str(s) for s in reactants_smiles):
            stoich['reaction_type'] = 'metal_oxide_oxidation'
            stoich['reactant_coeffs'] = {'CuO': 1, 'Organic': 1}
            stoich['product_coeffs'] = {'Oxidized': 1, 'H2O': 1}
        
        # Hydrolysis reactions
        elif 'O' in reactants_smiles:  # Water
            for r_smiles in reactants_smiles:
                if 'C#N' in r_smiles:  # Nitrile
                    stoich['reaction_type'] = 'nitrile_hydrolysis'
                    stoich['reactant_coeffs'] = {'RCN': 1, 'H2O': 2}
                    stoich['product_coeffs'] = {'RCOOH': 1, 'NH3': 1}
                    break
                elif 'N=C=S' in r_smiles:  # Isothiocyanate
                    stoich['reaction_type'] = 'isothiocyanate_hydrolysis'
                    stoich['reactant_coeffs'] = {'RNCS': 1, 'H2O': 1}
                    stoich['product_coeffs'] = {'RNH2': 1, 'COS': 1}
                    break
            else:
                stoich['reaction_type'] = 'general_hydrolysis'
                stoich['reactant_coeffs'] = {'Compound': 1, 'H2O': 1}
                stoich['product_coeffs'] = {'Hydrolyzed': 1}
        
        # Acetylation reactions
        elif 'CC(=O)O' in reactants_smiles:
            stoich['reaction_type'] = 'acetylation'
            stoich['reactant_coeffs'] = {'CH3COOH': 1, 'ROH': 1}
            stoich['product_coeffs'] = {'CH3COOR': 1, 'H2O': 1}
        
        else:
            # Default stoichiometry
            stoich['reaction_type'] = 'general'
            stoich['reactant_coeffs'] = {'Reactant': 1}
            stoich['product_coeffs'] = {'Product': 1}
        
        return stoich
    
    def format_balanced_equation(self, reactant_names: List[str], product_names: List[str], 
                                stoich: Dict[str, Any]) -> str:
        """Format a balanced chemical equation with coefficients"""
        
        # Build reactant side
        reactant_parts = []
        if stoich['reaction_type'] == 'metal_carboxylate_Al':
            reactant_parts.append("Al")
            reactant_parts.append(f"3 {reactant_names[1] if len(reactant_names) > 1 else 'RCOOH'}")
        elif stoich['reaction_type'] == 'metal_carboxylate_Cu':
            reactant_parts.append("Cu")
            reactant_parts.append(f"2 {reactant_names[1] if len(reactant_names) > 1 else 'RCOOH'}")
        elif stoich['reaction_type'] == 'nitrile_hydrolysis':
            reactant_parts.append(reactant_names[0] if reactant_names else 'RCN')
            reactant_parts.append("2 H₂O")
        elif stoich['reaction_type'] == 'metal_oxide_oxidation':
            if 'Al2O3' in stoich['reactant_coeffs']:
                reactant_parts.append("Al₂O₃")
                reactant_parts.append(f"2 {reactant_names[1] if len(reactant_names) > 1 else 'Organic'}")
            else:
                reactant_parts.append("CuO")
                reactant_parts.append(reactant_names[1] if len(reactant_names) > 1 else 'Organic')
        else:
            # Default formatting
            for i, name in enumerate(reactant_names):
                coeff_key = list(stoich['reactant_coeffs'].keys())[i] if i < len(stoich['reactant_coeffs']) else None
                if coeff_key and stoich['reactant_coeffs'][coeff_key] != 1:
                    reactant_parts.append(f"{stoich['reactant_coeffs'][coeff_key]} {name}")
                else:
                    reactant_parts.append(name)
        
        # Build product side
        product_parts = []
        if stoich['reaction_type'] == 'metal_carboxylate_Al':
            product_parts.append("Al(RCOO)₃")
            product_parts.append("3/2 H₂")
        elif stoich['reaction_type'] == 'metal_carboxylate_Cu':
            product_parts.append("Cu(RCOO)₂")
            product_parts.append("H₂")
        elif stoich['reaction_type'] == 'nitrile_hydrolysis':
            product_parts.append(product_names[0] if product_names else 'RCOOH')
            product_parts.append("NH₃")
        else:
            # Default formatting
            for i, name in enumerate(product_names):
                coeff_key = list(stoich['product_coeffs'].keys())[i] if i < len(stoich['product_coeffs']) else None
                if coeff_key and stoich['product_coeffs'][coeff_key] != 1:
                    if stoich['product_coeffs'][coeff_key] == 1.5:
                        product_parts.append(f"3/2 {name}")
                    else:
                        product_parts.append(f"{stoich['product_coeffs'][coeff_key]} {name}")
                else:
                    product_parts.append(name)
        
        equation = " + ".join(reactant_parts) + " → " + " + ".join(product_parts)
        return equation
    
    def display_stoichiometry(self, reactant_names: List[str], product_names: List[str], 
                            reactants_smiles: List[str], products_smiles: List[str]) -> str:
        """Display stoichiometric information for the reaction"""
        
        # Identify metals
        metal_reactants = []
        for smiles in reactants_smiles:
            if '[Al]' in smiles:
                metal_reactants.append('Al')
            elif '[Cu]' in smiles:
                metal_reactants.append('Cu')
            elif '[Ni]' in smiles:
                metal_reactants.append('Ni')
        
        # Calculate stoichiometry
        stoich = self.calculate_stoichiometry(reactants_smiles, products_smiles, metal_reactants)
        
        # Format equation
        balanced_eq = self.format_balanced_equation(reactant_names, product_names, stoich)
        
        print("\n" + "-"*40)
        print("STOICHIOMETRY")
        print("-"*40)
        print(f"Reaction Type: {stoich['reaction_type'].replace('_', ' ').title()}")
        print(f"Balanced Equation:")
        print(f"  {balanced_eq}")
        
        # Add molar ratios
        if stoich['reaction_type'] in ['metal_carboxylate_Al', 'metal_carboxylate_Cu']:
            print("\nMolar Ratios:")
            if 'Al' in metal_reactants:
                print("  1 mol Al : 3 mol acid : 1 mol Al(RCOO)₃ : 1.5 mol H₂")
            elif 'Cu' in metal_reactants:
                print("  1 mol Cu : 2 mol acid : 1 mol Cu(RCOO)₂ : 1 mol H₂")
        
        return balanced_eq
    
    def calculate_rate_constant(self, temperature: float, condition: str) -> Tuple[float, int]:
        """Calculate reaction rate constant using Arrhenius equation"""
        params = self.condition_params.get(condition, self.condition_params["ideal"])
        A = params["A"]
        Ea = params["Ea"]
        
        k = A * np.exp(-Ea / (self.R * temperature))
        return k, params["order"]
    
    def calculate_rate_constant(self, temperature: float, condition: str) -> Tuple[float, int]:
        """Calculate reaction rate constant using Arrhenius equation"""
        params = self.condition_params.get(condition, self.condition_params["ideal"])
        A = params["A"]
        Ea = params["Ea"]
        
        k = A * np.exp(-Ea / (self.R * temperature))
        return k, params["order"]
    
    def simulate_reaction(self, initial_conc: float, k: float, time_points: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate reaction progress over time"""
        if order == 0:
            reactant_conc = np.maximum(initial_conc - k * time_points, 0)
        elif order == 1:
            reactant_conc = initial_conc * np.exp(-k * time_points)
        elif order == 2:
            reactant_conc = initial_conc / (1 + initial_conc * k * time_points)
        else:
            # General case using numerical integration
            def rate_eq(t, C):
                return -k * (max(C[0], 0) ** order)
            
            sol = solve_ivp(rate_eq, [0, time_points[-1]], [initial_conc], 
                          t_eval=time_points, method='RK45')
            reactant_conc = np.maximum(sol.y[0], 0)
        
        product_conc = initial_conc - reactant_conc
        return reactant_conc, product_conc
    
    def plot_reaction_progress(self, time_points: np.ndarray, reactant_conc: np.ndarray, 
                              product_conc: np.ndarray, reactant_names: List[str], 
                              temperature: float, condition: str, k: float, order: int) -> str:
        """Plot reaction progress"""
        plt.figure(figsize=(12, 8))
        
        # Convert time to appropriate units
        if time_points[-1] > 3600:
            time_display = time_points / 3600
            time_label = 'Time (hours)'
        elif time_points[-1] > 60:
            time_display = time_points / 60
            time_label = 'Time (minutes)'
        else:
            time_display = time_points
            time_label = 'Time (seconds)'
        
        plt.plot(time_display, reactant_conc, 'b-', linewidth=2, 
                label=f"{', '.join(reactant_names)} (Reactants)")
        plt.plot(time_display, product_conc, 'r-', linewidth=2, 
                label="Products")
        
        # Add half-life for first-order reactions
        if order == 1 and k > 0:
            t_half = np.log(2) / k
            if t_half <= time_points[-1]:
                t_half_display = t_half / 3600 if time_points[-1] > 3600 else (t_half / 60 if time_points[-1] > 60 else t_half)
                plt.axvline(x=t_half_display, color='g', linestyle='--', 
                          label=f"t½ = {t_half_display:.2f}")
        
        plt.xlabel(time_label, fontsize=12)
        plt.ylabel('Concentration (mol/L)', fontsize=12)
        plt.title(f'Reaction Progress at {temperature}K ({condition})\n' + 
                 f'k = {k:.2e}, order = {order}', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = "reaction_progress.png"
        plt.savefig(filename, dpi=150)
        plt.show()
        
        return filename
    
    def run_complete_analysis(self, reactant_names: List[str], temperature: float = 298.15, 
                            simulation_time: float = 86400, condition: str = None,
                            user_products: List[str] = None) -> Dict[str, Any]:
        """Run complete reaction analysis"""
        results = {
            'reactants': {},
            'products': {},
            'kinetics': {},
            'stoichiometry': {},
            'visualizations': []
        }
        
        print("\n" + "="*60)
        print("REACTION ANALYSIS")
        print("="*60)
        
        # Get SMILES for reactants
        reactants_smiles = []
        valid_reactant_names = []
        
        for name in reactant_names:
            print(f"\nProcessing reactant: {name}")
            smiles = self.get_smiles_from_name(name)
            if smiles:
                reactants_smiles.append(smiles)
                valid_reactant_names.append(name)
                results['reactants'][name] = smiles
                print(f"✓ SMILES: {smiles}")
            else:
                print(f"✗ Could not find SMILES for '{name}'")
        
        if not reactants_smiles:
            print("\nError: No valid reactants found!")
            return results
        
        # Analyze reaction type
        print("\n" + "-"*40)
        print("REACTION TYPE ANALYSIS")
        print("-"*40)
        
        reaction_analysis = self.analyze_reaction_type(reactants_smiles)
        
        print(f"Reaction class: {reaction_analysis['reaction_class']}")
        if reaction_analysis['metals']:
            print(f"Metals detected: {', '.join(reaction_analysis['metals'])}")
        if reaction_analysis['functional_groups']:
            print(f"Functional groups: {', '.join(reaction_analysis['functional_groups'])}")
        
        # Use provided condition or auto-select
        if not condition:
            # Use auto-selected condition
            condition = reaction_analysis.get('auto_condition', 'ideal')
            print(f"\nAuto-selected condition: {condition}")
            
            if reaction_analysis['suggested_conditions']:
                print(f"Alternative conditions: {', '.join(reaction_analysis['suggested_conditions'])}")
        else:
            print(f"\nUsing user-specified condition: {condition}")
        
        # Predict products
        print("\n" + "-"*40)
        print("PRODUCT PREDICTION")
        print("-"*40)
        
        if user_products:
            # Use user-provided products
            products_smiles = []
            for name in user_products:
                smiles = self.get_smiles_from_name(name)
                if smiles:
                    products_smiles.append(smiles)
                    results['products'][name] = smiles
                    print(f"User product '{name}': {smiles}")
        else:
            # Predict products
            products_smiles = self.predict_products(reactants_smiles, reaction_analysis)
            for i, smiles in enumerate(products_smiles):
                # Give meaningful names to products
                if '[Al+3]' in smiles and 'C(=O)[O-]' in smiles:
                    name = "Aluminum_carboxylate"
                elif '[Cu+2]' in smiles and 'C(=O)[O-]' in smiles:
                    name = "Copper_carboxylate"
                elif '[H][H]' == smiles:
                    name = "Hydrogen_gas"
                elif '[Al+3].[OH-]' in smiles:
                    name = "Aluminum_hydroxide"
                elif '[Cu+2].[OH-]' in smiles:
                    name = "Copper_hydroxide"
                else:
                    name = f"Product_{i+1}"
                
                results['products'][name] = smiles
                print(f"Predicted {name}: {smiles}")
        
        # Calculate and display stoichiometry
        product_names = list(results['products'].keys())
        balanced_equation = self.display_stoichiometry(
            valid_reactant_names, 
            product_names,
            reactants_smiles,
            products_smiles
        )
        
        results['stoichiometry'] = {
            'balanced_equation': balanced_equation,
            'reactant_names': valid_reactant_names,
            'product_names': product_names
        }
        
        # Calculate kinetics
        print("\n" + "-"*40)
        print("KINETICS CALCULATION")
        print("-"*40)
        
        k, order = self.calculate_rate_constant(temperature, condition)
        results['kinetics'] = {
            'temperature': temperature,
            'condition': condition,
            'rate_constant': k,
            'order': order,
            'simulation_time': simulation_time
        }
        
        print(f"Temperature: {temperature} K")
        print(f"Condition: {condition}")
        print(f"Rate constant (k): {k:.2e}")
        print(f"Reaction order: {order}")
        
        # Simulate reaction
        time_points = np.linspace(0, simulation_time, 500)
        initial_conc = 1.0
        reactant_conc, product_conc = self.simulate_reaction(initial_conc, k, time_points, order)
        
        conversion = (1 - reactant_conc[-1]/initial_conc) * 100
        print(f"Conversion after {simulation_time/3600:.1f} hours: {conversion:.1f}%")
        
        results['kinetics']['conversion'] = conversion
        
        # Create visualizations
        print("\n" + "-"*40)
        print("CREATING VISUALIZATIONS")
        print("-"*40)
        
        # Plot reaction progress
        plot_file = self.plot_reaction_progress(
            time_points, reactant_conc, product_conc,
            valid_reactant_names, temperature, condition, k, order
        )
        if plot_file:
            results['visualizations'].append(plot_file)
            print(f"✓ Reaction progress plot: {plot_file}")
        
        # Create molecular visualizations
        for name, smiles in results['reactants'].items():
            # Skip bare metal atoms for structure visualization
            if smiles not in ['[Al]', '[Cu]', '[Ni]', '[Fe]', '[Pt]', '[Pd]']:
                img, props = self.visualizer.create_enhanced_2d_structure(smiles, name)
                if img:
                    filename = f"{name.replace(' ', '_')}_structure.png"
                    img.save(filename)
                    results['visualizations'].append(filename)
                    print(f"✓ Reactant structure: {filename}")
            else:
                # Create simple metal representation
                filename = f"{name.replace(' ', '_')}_structure.png"
                img = Image.new('RGB', (200, 200), 'white')
                draw = ImageDraw.Draw(img)
                draw.ellipse([70, 70, 130, 130], outline="black", width=2)
                draw.text((85, 85), smiles.strip('[]'), fill="black")
                img.save(filename)
                results['visualizations'].append(filename)
                print(f"✓ Reactant structure: {filename}")
        
        for name, smiles in results['products'].items():
            # Skip complex ionic representations
            if not ('[' in smiles and '+' in smiles and '.' in smiles):
                img, props = self.visualizer.create_enhanced_2d_structure(smiles, name)
                if img:
                    filename = f"{name.replace(' ', '_')}_structure.png"
                    img.save(filename)
                    results['visualizations'].append(filename)
                    print(f"✓ Product structure: {filename}")
        
        # Create reaction mechanism visualization
        mechanism_file = self.visualizer.visualize_reaction_mechanism(
            list(results['reactants'].values()),
            list(results['products'].values())
        )
        if mechanism_file:
            results['visualizations'].append(mechanism_file)
            print(f"✓ Reaction mechanism: {mechanism_file}")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        return results


def interactive_mode():
    """Interactive command-line interface with Excel upload support"""
    predictor = ReactionPredictor()
    
    print("\n" + "="*70)
    print(" ENHANCED CHEMICAL REACTION PREDICTOR ".center(70))
    print("="*70)
    print("\nThis system uses multiple chemical databases for compound lookup")
    print("and supports both organic and organometallic reactions.\n")
    
    while True:
        print("\n" + "-"*50)
        print("INPUT METHOD SELECTION")
        print("-"*50)
        print("\n1. Upload Excel file with reactions")
        print("2. Type reactant names manually")
        print("3. Exit program")
        
        choice = input("\nSelect input method (1-3): ").strip()
        
        if choice == '3':
            print("\nThank you for using the Chemical Reaction Predictor!")
            break
        
        elif choice == '1':
            # Excel file processing
            results = process_excel_file(predictor)
            if results:
                display_results_summary(results)
        
        elif choice == '2':
            # Manual input processing
            results = process_manual_input(predictor)
            if results:
                display_results_summary(results)
        
        else:
            print("Invalid choice. Please select 1, 2, or 3.")
        
        # Ask to continue
        print("\nWould you like to analyze more reactions? (y/n):")
        if input("> ").strip().lower() not in ['y', 'yes']:
            print("\nThank you for using the Chemical Reaction Predictor!")
            break


def process_excel_file(predictor):
    """Process reactions from an Excel file"""
    print("\n" + "-"*40)
    print("EXCEL FILE PROCESSING")
    print("-"*40)
    
    filename = input("\nEnter Excel filename (with .xlsx extension): ").strip()
    
    if not filename:
        print("No filename provided.")
        return None
    
    if not filename.endswith('.xlsx'):
        filename += '.xlsx'
    
    try:
        import pandas as pd
        
        # Read Excel file
        print(f"\nReading {filename}...")
        df = pd.read_excel(filename)
        
        print(f"Found {len(df)} rows in the Excel file.")
        print(f"Columns: {', '.join(df.columns)}")
        
        # Show preview
        print("\nFirst 5 rows of data:")
        print(df.head().to_string())
        
        # Determine column structure
        print("\n" + "-"*40)
        print("COLUMN MAPPING")
        print("-"*40)
        print("\nPlease identify the columns:")
        
        columns = list(df.columns)
        for i, col in enumerate(columns, 1):
            print(f"{i}. {col}")
        
        # Get reactant columns
        reactant_cols = []
        print("\nWhich columns contain reactants? (comma-separated numbers, or 'auto' to auto-detect):")
        reactant_input = input("> ").strip()
        
        if reactant_input.lower() == 'auto':
            # Auto-detect based on common patterns
            for col in columns:
                col_lower = col.lower()
                if 'reactant' in col_lower or 'reagent' in col_lower or 'substrate' in col_lower:
                    reactant_cols.append(col)
                elif any(metal in col_lower for metal in ['al', 'cu', 'ni', 'metal']):
                    reactant_cols.append(col)
            
            if not reactant_cols:
                # Default to first column if no pattern matches
                reactant_cols = [columns[0]]
            
            print(f"Auto-detected reactant columns: {', '.join(reactant_cols)}")
        else:
            try:
                indices = [int(i.strip()) - 1 for i in reactant_input.split(',')]
                reactant_cols = [columns[i] for i in indices if 0 <= i < len(columns)]
            except:
                print("Invalid input. Using first column as reactant.")
                reactant_cols = [columns[0]]
        
        # Get product columns (optional)
        product_cols = []
        print("\nWhich columns contain products? (comma-separated numbers, or press Enter to skip):")
        product_input = input("> ").strip()
        
        if product_input:
            try:
                indices = [int(i.strip()) - 1 for i in product_input.split(',')]
                product_cols = [columns[i] for i in indices if 0 <= i < len(columns)]
            except:
                print("Invalid input. Products will be predicted.")
        
        # Get condition column (optional)
        print("\nWhich column contains reaction conditions? (number, or press Enter for auto-detection):")
        condition_input = input("> ").strip()
        
        condition_col = None
        if condition_input:
            try:
                idx = int(condition_input) - 1
                if 0 <= idx < len(columns):
                    condition_col = columns[idx]
            except:
                pass
        
        # Process reactions
        print("\n" + "-"*40)
        print("PROCESSING REACTIONS")
        print("-"*40)
        
        all_results = []
        rows_to_process = min(len(df), 10)  # Limit to first 10 for demo
        
        print(f"\nProcessing first {rows_to_process} reactions...")
        
        for idx, row in df.head(rows_to_process).iterrows():
            print(f"\n--- Reaction {idx + 1} ---")
            
            # Extract reactants
            reactants = []
            for col in reactant_cols:
                if pd.notna(row[col]):
                    value = str(row[col]).strip()
                    # Parse multiple reactants if separated by + or ,
                    if '+' in value:
                        reactants.extend([r.strip() for r in value.split('+')])
                    elif ',' in value:
                        reactants.extend([r.strip() for r in value.split(',')])
                    else:
                        reactants.append(value)
            
            if not reactants:
                print("No reactants found in this row. Skipping.")
                continue
            
            print(f"Reactants: {', '.join(reactants)}")
            
            # Extract products if available
            products = []
            for col in product_cols:
                if pd.notna(row[col]):
                    value = str(row[col]).strip()
                    if '+' in value:
                        products.extend([p.strip() for p in value.split('+')])
                    elif ',' in value:
                        products.extend([p.strip() for p in value.split(',')])
                    else:
                        products.append(value)
            
            if products:
                print(f"Expected products: {', '.join(products)}")
            
            # Extract condition if available
            condition = None
            if condition_col and pd.notna(row[condition_col]):
                condition = str(row[condition_col]).strip().lower()
                # Map common condition names
                condition_map = {
                    'heat': 'pyrolysis',
                    'burn': 'combustion',
                    'oxidize': 'oxidation',
                    'reduce': 'reduction',
                    'metal': 'metal_coordination',
                }
                for key, value in condition_map.items():
                    if key in condition:
                        condition = value
                        break
            
            # Get temperature (default or from user)
            temperature = 298.15  # Default room temperature
            
            # Run analysis
            try:
                results = predictor.run_complete_analysis(
                    reactants,
                    temperature=temperature,
                    simulation_time=3600,  # 1 hour default
                    condition=condition,
                    user_products=products if products else None
                )
                all_results.append(results)
                
                # Show brief summary
                if results['products']:
                    print(f"Predicted products: {', '.join(results['products'].keys())}")
                if results['kinetics']:
                    print(f"Conversion: {results['kinetics'].get('conversion', 0):.1f}%")
                
            except Exception as e:
                print(f"Error processing reaction: {e}")
                continue
        
        print("\n" + "="*50)
        print(f"COMPLETED PROCESSING {len(all_results)} REACTIONS")
        print("="*50)
        
        # Option to save results
        print("\nWould you like to save the results to a file? (y/n):")
        if input("> ").strip().lower() in ['y', 'yes']:
            save_results_to_file(all_results, f"results_{filename.replace('.xlsx', '')}.txt")
        
        return all_results
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None


def process_manual_input(predictor):
    """Process manually entered reactions"""
    print("\n" + "-"*40)
    print("MANUAL REACTION INPUT")
    print("-"*40)
    
    # Get reactants
    print("\nEnter reactant names (separate multiple with commas):")
    print("Examples: benzene, phenol | nickel, syringol | aluminum oxide, canthine-6-one")
    reactant_input = input("> ").strip()
    
    if not reactant_input:
        print("No input provided.")
        return None
    
    reactants = [r.strip() for r in reactant_input.split(',')]
    
    # Get optional products
    print("\nEnter product names if known (optional - press Enter to predict):")
    product_input = input("> ").strip()
    
    products = None
    if product_input:
        products = [p.strip() for p in product_input.split(',')]
    
    # Get reaction condition
    print("\nSelect reaction condition:")
    print("1. Ideal (default)")
    print("2. Pyrolysis (high temperature)")
    print("3. Combustion")
    print("4. Electrochemical")
    print("5. Metal reduction")
    print("6. Metal oxidation")
    print("7. Hydrolysis")
    print("8. Cross-coupling")
    print("9. Custom (enter name)")
    
    condition_choice = input("\nEnter choice (1-9): ").strip()
    
    condition_map = {
        '1': 'ideal',
        '2': 'pyrolysis',
        '3': 'combustion',
        '4': 'electrochemical',
        '5': 'metal_reduction',
        '6': 'metal_oxidation',
        '7': 'hydrolysis',
        '8': 'cross_coupling',
    }
    
    if condition_choice in condition_map:
        condition = condition_map[condition_choice]
    elif condition_choice == '9':
        condition = input("Enter custom condition name: ").strip().lower()
    else:
        condition = 'ideal'
    
    print(f"\nSelected condition: {condition}")
    
    # Get temperature
    print("\nEnter temperature:")
    print("1. Room temperature (298 K)")
    print("2. Boiling water (373 K)")
    print("3. Pyrolysis temperature (773 K)")
    print("4. Custom temperature")
    
    temp_choice = input("\nEnter choice (1-4): ").strip()
    
    temp_map = {
        '1': 298.15,
        '2': 373.15,
        '3': 773.15,
    }
    
    if temp_choice in temp_map:
        temperature = temp_map[temp_choice]
    elif temp_choice == '4':
        try:
            temperature = float(input("Enter temperature in Kelvin: ").strip())
        except:
            temperature = 298.15
    else:
        temperature = 298.15
    
    print(f"Temperature: {temperature} K")
    
    # Get simulation time
    print("\nEnter simulation time:")
    print("1. 1 minute (60 s)")
    print("2. 1 hour (3600 s)")
    print("3. 24 hours (86400 s)")
    print("4. Custom time")
    
    time_choice = input("\nEnter choice (1-4): ").strip()
    
    time_map = {
        '1': 60,
        '2': 3600,
        '3': 86400,
    }
    
    if time_choice in time_map:
        simulation_time = time_map[time_choice]
    elif time_choice == '4':
        try:
            simulation_time = float(input("Enter time in seconds: ").strip())
        except:
            simulation_time = 3600
    else:
        simulation_time = 3600
    
    print(f"Simulation time: {simulation_time} seconds")
    
    # Run analysis
    try:
        results = predictor.run_complete_analysis(
            reactants,
            temperature=temperature,
            simulation_time=simulation_time,
            condition=condition,
            user_products=products
        )
        return results
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        return None


def display_results_summary(results):
    """Display a summary of analysis results"""
    if not results:
        return
    
    # Handle both single result and list of results
    if isinstance(results, list):
        print("\n" + "="*50)
        print("BATCH RESULTS SUMMARY")
        print("="*50)
        
        for i, result in enumerate(results, 1):
            print(f"\nReaction {i}:")
            display_single_result(result)
    else:
        display_single_result(results)


def display_single_result(result):
    """Display a single reaction result"""
    if result.get('reactants'):
        print("  Reactants:")
        for name, smiles in result['reactants'].items():
            print(f"    - {name}: {smiles}")
    
    if result.get('products'):
        print("  Products:")
        for name, smiles in result['products'].items():
            print(f"    - {name}: {smiles}")
    
    if result.get('kinetics'):
        k = result['kinetics']
        print(f"  Kinetics:")
        print(f"    - Rate constant: {k['rate_constant']:.2e}")
        print(f"    - Order: {k['order']}")
        print(f"    - Conversion: {k.get('conversion', 0):.1f}%")
    
    if result.get('visualizations'):
        print(f"  Generated {len(result['visualizations'])} visualization files")


def save_results_to_file(results, filename):
    """Save analysis results to a text file"""
    try:
        with open(filename, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CHEMICAL REACTION ANALYSIS RESULTS\n")
            f.write("="*70 + "\n\n")
            
            if isinstance(results, list):
                for i, result in enumerate(results, 1):
                    f.write(f"Reaction {i}:\n")
                    f.write("-"*40 + "\n")
                    
                    if result.get('reactants'):
                        f.write("Reactants:\n")
                        for name, smiles in result['reactants'].items():
                            f.write(f"  {name}: {smiles}\n")
                    
                    if result.get('products'):
                        f.write("Products:\n")
                        for name, smiles in result['products'].items():
                            f.write(f"  {name}: {smiles}\n")
                    
                    if result.get('kinetics'):
                        k = result['kinetics']
                        f.write("Kinetics:\n")
                        f.write(f"  Temperature: {k['temperature']} K\n")
                        f.write(f"  Condition: {k['condition']}\n")
                        f.write(f"  Rate constant: {k['rate_constant']:.2e}\n")
                        f.write(f"  Order: {k['order']}\n")
                        f.write(f"  Conversion: {k.get('conversion', 0):.1f}%\n")
                    
                    f.write("\n")
            else:
                # Single result
                save_results_to_file([results], filename)
                return
        
        print(f"Results saved to {filename}")
        
    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    interactive_mode()

