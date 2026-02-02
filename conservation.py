from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from swift.utils import get_logger

logger = get_logger()


def get_mol_properties(mol):
    if mol is None:
        return defaultdict(int), 0, 0

    heavy_atoms = defaultdict(int)
    h_count = 0

    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 1:
            h_count += 1
        else:
            symbol = atom.GetSymbol()
            heavy_atoms[symbol] += 1

        h_count += atom.GetTotalNumHs()

    charge = Chem.rdmolops.GetFormalCharge(mol)

    return heavy_atoms, h_count, charge


def check_conservation(reaction_smiles):
    results = {
        'heavy_atom': False,
        'proton': False,
        'electron': False,
        'cumulative': False,
        'valid_reaction': True
    }

    try:
        rxn = rdChemReactions.ReactionFromSmarts(reaction_smiles, useSmiles=True)
        rdChemReactions.SanitizeRxn(rxn)
    except Exception as e:
        results['valid_reaction'] = False
        return results

    reactants_heavy_atoms = defaultdict(int)
    reactants_h_total = 0
    reactants_charge_total = 0

    for reactant_mol in rxn.GetReactants():
        reactant_mol.UpdatePropertyCache(strict=False)
        mol = Chem.AddHs(reactant_mol)
        heavy, h, charge = get_mol_properties(mol)
        for atom, count in heavy.items():
            reactants_heavy_atoms[atom] += count
        reactants_h_total += h
        reactants_charge_total += charge

    products_heavy_atoms = defaultdict(int)
    products_h_total = 0
    products_charge_total = 0

    for product_mol in rxn.GetProducts():
        product_mol.UpdatePropertyCache(strict=False)
        mol = Chem.AddHs(product_mol)
        heavy, h, charge = get_mol_properties(mol)
        for atom, count in heavy.items():
            products_heavy_atoms[atom] += count
        products_h_total += h
        products_charge_total += charge

    results['heavy_atom'] = (reactants_heavy_atoms == products_heavy_atoms)

    results['proton'] = (reactants_h_total == products_h_total)

    results['electron'] = (reactants_charge_total == products_charge_total)

    results['cumulative'] = all([results['heavy_atom'], results['proton'], results['electron']])

    return results


def calculate_conservation_rates(reaction_list):
    counts = defaultdict(int)
    valid_reactions = 0

    for rxn_smi in reaction_list:
        result = check_conservation(rxn_smi)
        if result['valid_reaction']:
            valid_reactions += 1
            for key, is_conserved in result.items():
                if is_conserved and key != 'valid_reaction':
                    counts[key] += 1

    if valid_reactions == 0:
        return {key: 0.0 for key in counts}

    rates = {key: (value / valid_reactions) * 100 for key, value in counts.items()}
    return rates
