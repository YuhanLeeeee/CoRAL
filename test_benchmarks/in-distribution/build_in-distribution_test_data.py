from tqdm import tqdm
from rdkit import Chem
import json
import random

with open("../../SFT_dataset/mechanism_template2.jsonl", 'r') as f:
    templates = [json.loads(line) for line in f]

with open("../../SFT_dataset/SFT_mechanism_stage0/filtered_mechanism_val_flower.jsonl", 'r') as f:  # 此处从训练数据中读取！
    data = [json.loads(line) for line in f]

# NEPP
save_data = []
tmp_list = [tmp for tmp in templates if tmp["task"] == 'nepp']

reaction_step = """
    \"Elementary Step <i>\": {
        "reactants": \"<reactants>\",
        "products": \"<products>\",
        "reaction_fingerprints": \"<reaction_fingerprints>\"
    }\n"""
answer_tmp = """
    {
      "Elementary Step <i>": {
        "reactants": "<reactants>",
        "products": "<products>",
      }
    }
    """
for reaction in tqdm(data):
    local_save_data = []
    for idx in range(1, len(reaction) + 1):
        question_data = {}
        tmp = random.choice(tmp_list)["template"]
        smile_data = []
        question = tmp.replace('<reactants>', '.'.join(reaction['1']['reactants'])) \
            .replace('<current_SMILES>', '.'.join(reaction[str(idx)]['reactants']))
        answer = answer_tmp.replace('<reactants>', '.'.join(reaction[str(idx)]['reactants']))
        random.shuffle(reaction['1']['reactants'])
        random.shuffle(reaction[str(idx)]['reactants'])
        reaction_steps = ""
        for i in range(1, idx):
            random.shuffle(reaction[str(i)]['reactants'])
            random.shuffle(reaction[str(i)]['products'])
            reaction_steps += reaction_step \
                .replace('<reactants>', '.'.join(reaction[str(i)]['reactants'])) \
                .replace('<products>', '.'.join(reaction[str(i)]['products'])).replace("<i>", str(i)) \
                .replace("<reaction_fingerprints>", "<FP_TOKEN>" * 64)
            smile_data.append({
                "idx": i,
                "reactants": '.'.join(reaction[str(i)]['reactants']),
                "products": '.'.join(reaction[str(i)]['products'])
            })
        question = question.replace("<reaction_step>", reaction_steps)
        question_data['question-<reaction_step>'] = reaction_steps
        answer = answer.replace('<i>', str(idx)).replace("<products>", '.'.join(reaction[str(idx)]['products']))
        question_data['answer'] = answer
        question_data['SMILES'] = smile_data
        question_data['task_type'] = 'nepp'
        message = {
            "messages": [{
                    "role": 'user',
                    "content": question + " /no_think"
                }],
            # "query": question + " /no_think",
            "task_type": "nepp",
            "SMILES": smile_data,
            "gt": '.'.join(reaction[str(idx)]['products'])
        }
        # save_data.append(message)
        local_save_data.append(message)
        # print(message)
        # print(("-----------------------------"))
    # exit()
    save_data.extend(random.sample(local_save_data, int(len(reaction) / 2)))
random.shuffle(save_data)
with open("./USPTO_test_nepp.jsonl", 'w', encoding='UTF-8') as f:
    for line in save_data:
        json_line = json.dumps(line)
        f.write(json_line + '\n')

# RP
save_data = []
tmp = [tmp for tmp in templates if tmp["task"] == 'fw'][0]['template']
for idx, reaction in tqdm(enumerate(data), total=len(data)):
    # print(reaction)
    question = tmp.replace('<reactants>', '.'.join(reaction['1']['reactants']))
    message = {
        "messages": [
            {
                "role": 'user',
                "content": question
            }
        ],
        "task_type": "fw",
        "gt": '.'.join(reaction[f'{len(reaction)}']['products']),
        "idx": idx
    }
    save_data.append(message)
random.shuffle(save_data)
with open("./USPTO_test_fs.jsonl", 'w') as f:
    for data in save_data:
        f.write(json.dumps(data) + '\n')
