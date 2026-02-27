import os
from utils import format_prediction_output

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
from swift.llm import get_model_tokenizer, get_template
from swift.utils import get_logger
import torch
from codebook.codebook_train import model as fp_model
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import Chem
from swift.llm import RequestConfig, InferRequest
from customized_swift.swift_ptengine import PtEngine

temperature = 0.3
count = 1
num_beam = 1
max_length = 4096


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


inference_data_path = './test_benchmarks/chemcotbench/nepp-fp.jsonl'
model_id_or_path = './checkpoints/CoRAL-8B'  # Qwen3-8B
fp_model_path = f"{model_id_or_path}/fp_model_params.pth"

fp_model.load_state_dict(torch.load(fp_model_path, map_location="cuda:0"))
fp_model = fp_model.to("cuda", dtype=torch.bfloat16)

# LLM
model, tokenizer = get_model_tokenizer(model_id_or_path, model_type='qwen3', torch_dtype=torch.bfloat16)
model = model.to("cuda")
logger = get_logger()
logger.info(f'model_info: {model.model_info}')
template = get_template(model.model_meta.template, tokenizer, max_length=max_length)
template.set_mode('vllm')

# load refer engine
engine = PtEngine.from_model_template(model, template)
engine.set_fp_model(fp_model)
request_config = RequestConfig(max_tokens=max_length, temperature=temperature)

with open(inference_data_path, 'r') as f:
    data = [json.loads(line) for line in f]

row = data[0]
messages = row['messages']
if "SMILES" in row.keys():
    messages[0]['fp'] = get_fp(row['SMILES'])
infer_requests = [InferRequest(messages=messages) for i in range(0, count)]
resp_list = engine.infer(infer_requests, request_config)
for resp in resp_list:
    predict = resp.choices[0].message.content
    format_prediction_output(predict)
