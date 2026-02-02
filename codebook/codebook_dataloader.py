import json

from tqdm import tqdm
import torch

from torch.utils.data import Dataset, DataLoader
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import Chem
from multiprocessing import Pool

torch.set_printoptions(profile="full")
torch.multiprocessing.set_sharing_strategy('file_system')


class CodeBookDataset(Dataset):
    def __init__(self):
        # with open("../SFT_dataset/SFT_mechanism_stage0/filtered_mechanism_train_sampled.jsonl", 'r') as f:
        f = open("../SFT_dataset/SFT_mechanism_stage0/filtered_mechanism_train_flower.jsonl", 'r')
        datas = [json.loads(line) for line in f]
        f.close()
        self.reaction_data = []
        with Pool(32) as pool:
            results = list(tqdm(
                pool.imap(self._process_single_reaction, datas),
                total=len(datas),
                desc="Processing tasks",
                unit="task"
            ))
        for reaction in results:
            for t in reaction:
                self.reaction_data.append(torch.tensor(t, dtype=torch.float))

    def _process_single_reaction(self, data):
        reaction_data = []
        morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)
        for k, reaction in data.items():
            reaction_fp = torch.zeros([1024], dtype=torch.float)
            for reactant in reaction['reactants']:
                mol = Chem.MolFromSmiles(reactant)
                fp_str = morgan_gen.GetFingerprint(mol).ToBitString()
                reaction_fp = torch.add(torch.tensor([int(bit) for bit in fp_str], dtype=torch.float), reaction_fp)
            product_fp = torch.zeros([1024], dtype=torch.float)
            for product in reaction['products']:
                mol = Chem.MolFromSmiles(product)
                fp_str = morgan_gen.GetFingerprint(mol).ToBitString()
                product_fp = torch.add(torch.tensor([int(bit) for bit in fp_str], dtype=torch.float), product_fp)
            fp = torch.concat((reaction_fp, product_fp), dim=-1)
            fp = fp.tolist()
            reaction_data.append(fp)

        del reaction_fp
        del product_fp
        return reaction_data

    def __len__(self):
        return len(self.reaction_data)

    def __getitem__(self, idx):
        fp = self.reaction_data[idx]
        return fp
