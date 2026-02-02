# Constructing prospective test data using Reaxys data from January 26
import json

import pandas as pd
import requests
from tqdm import tqdm
from rdkit import Chem

data_name = "rxn4000"
data = pd.read_excel(f"./{data_name}.xlsx")

reaction, reagent, solvent = data['Reaction'], data['Reagent'], data['Solvent (Reaction Details)']
yld = data["Yield (numerical)"]
print(len(reaction))

rgt_mp, sol_mp = [], []

file_path = f'./{data_name}_conditions.json'
with open(file_path, 'r', encoding='UTF8') as f:
    mp = json.load(f)
print(len(mp.keys()))

petasis_reactions = []


def remove_atom_mapping(smiles):
    # print(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise Exception

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(mol)


reactions = {}
for rct, rgt, sol, y in zip(reaction, reagent, solvent, yld):
    if type(rct) == float:
        continue
    reactants, products = rct.split('>>')
    _rgt, _sol = [], []
    flag = True
    if type(rgt) != float:
        for r in rgt.split(';'):
            r = r.strip()
            if r not in mp.keys() or mp[r] == "":
                flag = False
                continue
            _rgt.append(remove_atom_mapping(mp[r][1:-1]))
    if type(sol) != float:
        for s in sol.split(';'):
            s = s.strip()
            if s not in mp.keys() or mp[s] == "":
                flag = False
                continue
            _sol.append(remove_atom_mapping(mp[s][1:-1]))
    try:
        react = '.'.join([remove_atom_mapping(smiles) for smiles in reactants.split('.')])
        products = '.'.join([remove_atom_mapping(smiles) for smiles in products.split('.')])
    except:
        continue
    simple_reaction = react + ">>" + products
    if len(_rgt) != 0:
        react += '.' + '.'.join(_rgt)
    if len(_sol) != 0:
        react += '.' + '.'.join(_sol)
    react += '>>' + products
    if flag:
        try:
            yy = float(y)
            if simple_reaction not in reactions.keys():
                reactions[simple_reaction] = (react, yy)
            elif float(reactions[simple_reaction][1]) > yy:
                reactions[simple_reaction] = (react, yy)
        except:
            continue

print(len(reactions.keys()))
with open(f"./{data_name}.txt", 'w', encoding='UTF8') as f:
    for k, v in reactions.items():
        f.write(v[0] + '\n')
print("save!")
exit()

# Convert reagents and solvents from names to SMILES
names = sol_mp + rgt_mp
base_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES,title/CSV'
headers = {"content-type": "application/x-www-form-urlencoded"}

results = {}
for name in tqdm(names, total=len(names)):
    try:
        url = base_url.format(name)
        res = requests.get(url, headers=headers)
        data_lines = res.text.strip().split('\n')[1:]
        if len(data_lines) != 1:
            results[name] = ""
            print("Not Found")
            continue
        results[name] = data_lines[0].split(',')[1]
    except:
        pass
with open(f"./{data_name}_conditions.json", 'w', encoding='UTF8') as f:
    f.write(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False))
print("save!")
