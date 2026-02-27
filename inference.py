import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
from swift.llm import get_model_tokenizer, get_template
from swift.utils import get_logger
import torch

from codebook.codebook_train import model as fp_model

from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import Chem
from tqdm import tqdm
from swift.llm import RequestConfig, InferRequest
from customized_swift.swift_ptengine import PtEngine

import argparse

parser = argparse.ArgumentParser(description="parser")

parser.add_argument("-i", "--inference_data_path", type=str, default="./test_benchmarks/chemcotbench/nepp-fp.jsonl")
parser.add_argument("--model_path", type=str, default="./checkpoints/CoRAL-8B")
parser.add_argument("--output_dir", type=str, default="./outputs")
parser.add_argument("-t", "--temperature", type=float, default=0.3)
parser.add_argument("-cnt", "--count", type=int, default=3)
parser.add_argument("-n", "--num_beam", type=int, default=1)

args = parser.parse_args()


def get_fp(smiles_data):
    fp = []
    morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)
    for reaction in smiles_data:
        reaction_fp = torch.zeros([1024], dtype=torch.float)
        for reactant in reaction['reactants'].split('.'):
            mol = Chem.MolFromSmiles(reactant)
            fp_str = morgan_gen.GetFingerprint(mol).ToBitString()
            reaction_fp = torch.add(torch.tensor([int(bit) for bit in fp_str], dtype=torch.float), reaction_fp)
        product_fp = torch.zeros([1024], dtype=torch.float)
        for product in reaction['products'].split('.'):
            mol = Chem.MolFromSmiles(product)
            fp_str = morgan_gen.GetFingerprint(mol).ToBitString()
            product_fp = torch.add(torch.tensor([int(bit) for bit in fp_str], dtype=torch.float), product_fp)
        _fp = torch.concat((reaction_fp, product_fp), dim=-1)
        _fp = _fp.tolist()
        fp.append(_fp)
    return fp


inference_data_path = args.inference_data_path
model_id_or_path = args.model_path
fp_model_path = f"{model_id_or_path}/fp_model_params.pth"

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

model_name = model_id_or_path.split('/')[-1]
task = inference_data_path.split('/')[-1].split('.')[0]
print("task: ", task)
print("inference_data_path: ", inference_data_path)

save_path = f"{output_dir}/{model_name}-{task}-withfp-t{int(args.temperature * 10)}_repeat{int(args.count)}_beam{int(args.num_beam)}.jsonl"
print("save_path: ", save_path)

fp_model.load_state_dict(torch.load(fp_model_path, map_location="cuda:0"))
fp_model = fp_model.to("cuda", dtype=torch.bfloat16)

# CoRAL
model, tokenizer = get_model_tokenizer(model_id_or_path, model_type='qwen3', torch_dtype=torch.bfloat16)
model = model.to("cuda")
logger = get_logger()
logger.info(f'model_info: {model.model_info}')
template = get_template(model.model_meta.template, tokenizer, max_length=4096)
template.set_mode('vllm')

engine = PtEngine.from_model_template(model, template)
engine.set_fp_model(fp_model)
request_config = RequestConfig(max_tokens=4096, temperature=args.temperature)
with open(inference_data_path) as f:
    data = [json.loads(line) for line in f]
results = []

cnt = len(data)
right = 0.0
with open(save_path, 'w', encoding='UTF-8') as f:
    with tqdm(enumerate(data), total=len(data)) as bar:
        for idx, row in bar:
            messages = row['messages']
            if "SMILES" in row.keys():
                messages[0]['fp'] = get_fp(row['SMILES'])
            infer_requests = [InferRequest(messages=messages) for i in range(0, args.count)]
            resp_list = engine.infer(infer_requests, request_config)
            try:
                for resp in resp_list:
                    predict = resp.choices[0].message.content
                    result = {'response': predict, 'idx': idx}
                    results.append(result)

                    json_line = json.dumps(result)
                    f.write(json_line + '\n')
                    f.flush()
            except:
                continue
print(f"Save！ Total: {len(results)}, path: {save_path}")
