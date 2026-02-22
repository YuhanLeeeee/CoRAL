# CoRAL: Physically Grounded Multi-modal Reasoning for Organic Reaction Mechanism Prediction

This is the official repository for the paper **"Physically Grounded Multi-modal Reasoning for Organic Reaction Mechanism Prediction"**.

> ⚠️ **Note**
>
> **This repository is continuously being updated. We are actively refining the documentation and uploading more checkpoints and experimental data.**

---
## 📂 Repository Structure

The file structure and function of the key components are organized as follows:

```text
CoRAL/
├── checkpoints/          # Model weights and training checkpoints
├── codebook/             # Configuration and data related to the Codebook
├── customized_swift/     # Customized implementation of the SWIFT fine-tuning framework
├── test_benchmarks/      # Datasets and scripts for benchmark evaluation
├── yiled_prediction/     # Code for the "Yield Prediction" downstream task
├── SFT.py                # Main script for Supervised Fine-Tuning (SFT)
├── conservation.py       # Script for calculating conservation metrics
├── ds_config.json        # DeepSpeed configuration for distributed training
├── main.sh               # Entry point shell script to run the project
├── modify_tokenizer.py   # Utility to modify or extend the model tokenizer
└── README.md             # Project documentation
```

---
## ⚙️ System Requirements

### 1. Hardware Requirements
This project involves large-scale model training/inference and requires high-performance computing resources.

*   **Recommended Hardware:** 
    *   **GPU:** At least one NVIDIA GPU with **80GB VRAM** (or higher).
    *   **Architecture:** Ampere (A100/A800) or Hopper (H100/H800) recommended.
    *   **Tested Hardware:** NVIDIA A800 80GB PCIe

### 2. Software & Key Dependencies
The project relies on specific versions of deep learning frameworks. Below are the core dependencies:

*   **Python Version:** Python 3.10+ (Recommended)
*   **Core Libraries:**
    *   `ms_swift` == **3.7.3**
    *   `torch` == **2.7.0+cu118** (CUDA 11.8 build)
    *   `transformers` == **4.52.4**
    *   `deepspeed` == **0.17.2**
    *   `rdkit` == **2025.3.3**
    *   `tokenizers` == **0.22.2**

> **Note:** For a complete list of all dependencies, please refer to [requirements.txt](./requirements.txt).

### 3. Tested Environment
The software has been rigorously tested and verified on the following environment setup:

| Component | Version / Details |
| :--- | :--- |
| **OS** | Linux (Ubuntu 22.04) |
| **NVIDIA Driver** | **535.104.12** |
| **CUDA Version** | **12.2** (System-level) |
| **GPU** | NVIDIA A800 80GB PCIe |

---

## 🚀 Installation

To set up the environment with all the required dependencies, please follow these steps:

Follow these steps to set up the environment and install dependencies via the command line.

### **Step 1: Clone the repository**
```bash
git clone https://github.com/YuhanLeeeee/CoRAL.git
cd CoRAL
```

### **Step 2: Create a Virtual Environment (Recommended)**

Since this project requires specific versions of deep learning frameworks (PyTorch 2.7, etc.), we highly recommend using Conda to isolate the environment.

```bash
# Create a new environment with Python 3.10
conda create -n CoRAL python=3.10

# Activate the environment
conda activate CoRAL
```

### **Step 3: Install Dependencies**

Install all required packages listed in requirements.txt.

```bash
# Upgrade pip to ensure it can handle modern wheels
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

*Note: Since `torch` specifies a local version identifier (`+cu118`), ensure your pip allows version specifiers or use the extra index URL if necessary:*

```bash
pip install torch==2.7.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
### **Step 4: Verify Installation**

Run the following command to check if the environment is set up correctly:

```bash
python your_main_script.py --help
# Or check version
python -c "import torch; print(torch.__version__)"
```

### **Typical Install Time**

* Estimated Time: 15 - 30 minutes
* Factors:
  * Standard Desktop (8-core CPU, 16GB RAM): Approximately 20 minutes.
  * The installation time is primarily dependent on your internet speed, as the dependencies include large libraries (e.g., PyTorch ~2.5GB, CUDA kernels).
  * Some packages like deepspeed may require JIT compilation during the first run or installation, which can take a few extra minutes depending on your CPU.

---
## 🧪 Demo & Usage Examples

This section provides instructions on how to run the demos for **Next Elementary Product Prediction** and **Overall Reaction Prediction**. 

Ensure you have activated your environment before running these commands.

### 📥 Download Model Weights

Before running the demos, you need to download the pre-trained model weights from our Hugging Face repository: [🤗 **yhLi/CoRAL-8B**](https://huggingface.co/yhLi/CoRAL-8B)

You can download the models using Hugging Face CLI (Recommended):

Ensure `huggingface_hub` is installed (included in `requirements.txt`), then run:

```bash
# Download the model to the 'checkpoints' folder
huggingface-cli download --resume-download yhLi/CoRAL-8B --local-dir checkpoints/CoRAL-8B
```

### Case 1: Next Elementary Product Prediction (NEPP)

This demo predicts the immediate next elementary step in a reaction mechanism given a set of reactants.

**Run the demo:**
```bash
python demo_NEPP.py
```

**Expected Output:**

After running the script, the model will load the weights and output the predicted elementary reaction products based on the previously input elementary reaction steps and the reactants of the current elementary reaction:

```
================================================================================
                              ✨ Elementary Step 6
================================================================================

[INPUT REACTANTS]:
CCN=C=NCCCN(C)C.O=C(O)c1ccc(Oc2ccc(OCc3ccccc3)cc2)cc1

--------------------------------------------------------------------------------

[PREDICTED PRODUCTS]:
CCN=C=[NH+]CCCN(C)C.O=C([O-])c1ccc(Oc2ccc(OCc3ccccc3)cc2)cc1
================================================================================

```

**Expected Run Time:**

GPU (e.g., A800): < 10 seconds.

### Case 2: Reaction Prediction (RP)

This demonstration predicts the final products of chemical reaction sequences, as well as the overall elementary reaction pathways.

**Run the demo:**
```
python demo_RP.py
```

**Expected Output:**

The script will output the predicted final product SMILES string.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ⚛️  REACTION MECHANISM ANALYSIS REPORT                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Found 5 elementary steps in reasoning process...

▶ Elementary Step 1
  Input : Cc1ccc(C)c2c(C(C)C)cc(O)nc12.O=P(Cl)(Cl)Cl.Cc1ccccc1
  Output: Cc1ccccc1.Cc1ccc(C)c2c(C(C)C)cc([OH+]P([O-])(Cl)(Cl)Cl)nc12
      ↓ (proceeds to next step)
----------------------------------------
▶ Elementary Step 2
  Input : Cc1ccc(C)c2c(C(C)C)cc([OH+]P([O-])(Cl)(Cl)Cl)nc12.Cc1ccccc1
  Output: Cc1ccccc1.Cc1ccc(C)c2c(C(C)C)cc([OH+]P(=O)(Cl)Cl)nc12.[Cl-]
      ↓ (proceeds to next step)
----------------------------------------
▶ Elementary Step 3
  Input : Cc1ccccc1.Cc1ccc(C)c2c(C(C)C)cc([OH+]P(=O)(Cl)Cl)nc12.[Cl-]
  Output: Cc1ccccc1.[Cl-].Cc1ccc(C)c2c(C(C)C)cc(OP(=O)(Cl)Cl)nc12
      ↓ (proceeds to next step)
----------------------------------------
▶ Elementary Step 4
  Input : Cc1ccc(C)c2c(C(C)C)cc(OP(=O)(Cl)Cl)nc12.Cc1ccccc1.[Cl-]
  Output: O=P(=O)Cl.Cc1ccccc1.Cc1ccc(C)c2c(C(C)C)cc(Cl)nc12.[Cl-]
      ↓ (proceeds to next step)
----------------------------------------
▶ Elementary Step 5
  Input : O=P(=O)Cl.Cc1ccccc1.Cc1ccc(C)c2c(C(C)C)cc(Cl)nc12.[Cl-]
  Output: Cc1ccc(C)c2c(C(C)C)cc(Cl)nc12.Cc1ccccc1.[Cl-].O=P(=O)Cl

╔══════════════════════════════════════════════════════════════════════════════╗
║                          ✨ FINAL PREDICTION RESULT                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
      SMILES: Cc1ccc(C)c2c(C(C)C)cc(Cl)nc12.Cc1ccccc1.[Cl-].O=P(=O)Cl
╚══════════════════════════════════════════════════════════════════════════════╝
```

**Expected Run Time:**

GPU (e.g., A800): < 40 seconds.

---
## 🔬 Instructions for Use

This section describes how to run the model on your own datasets for batch inference.

### 1. Data Preparation

To evaluate or use the model on your custom data, you must format your input file as a **JSONL (JSON Lines)** file. Each line should be a valid JSON object containing the necessary fields.

Taking the **NEPP (Next Elementary Product Prediction)** task as an example, your dataset must strictly follow this structure:
*   `id` (optional): Unique identifier for the sample.
*   `SMILES`: The reaction SMILES string (used for fingerprint generation).
*   `messages`: A list containing the prompt structure compatible with the LLM (Qwen template).
*   `task_type`: Specified prediction task.

**Example format (`./test_benchmarks/chemcotbench/nepp-fp.jsonl`):**
```json
{"messages": [{"role": "user", "content": "Your task is to act as an experienced chemist and predict the products of the next elementary reaction step in an ongoing multi-step reaction. You will be given the overall starting materials and reagents for the entire reaction, a history of all previous elementary steps, and the reactants for the current step. Based on this complete context, you must deduce the products formed from the current reactants.\n\nInput:\nA JSON object containing the following three key pieces of information:\n1.overall_reactants: The SMILES strings for the initial reactants and reagents of the entire reaction, separated by ..\n2.previous_steps: A list of objects representing the elementary steps that have already occurred. Each object contains its reactants and products. This list can be empty (if this is the first step).\n3.current_reactants: The SMILES string(s) of the reactants for the current elementary step that you need to predict.\n\nOutput:\nYou must predict the products formed from the current_reactants under the given reaction context. The output SMILES must be valid and chemically reasonable. Do not provide any additional information beyond the requested SMILES string(s). The answer must be in a JSON format, containing only the reactants and products of the next step.\n\nQuestion:\nReaction Context and Current Step\n{\n  \"overall_reactants\": \"COC(=O)c1ccccc1-c1ccc(Cl)c(C(=O)NCC2(C)CCCCCC2)c1\",\n  \"previous_steps\": [\n    \"Elementary Step 1\": {\n        \"reactants\": \"O[K].COC(=O)c1ccccc1-c1ccc(Cl)c(C(=O)NCC2(C)CCCCCC2)c1\",\n        \"products\": \"COC([O-])(O)c1ccccc1-c1ccc(Cl)c(C(=O)NCC2(C)CCCCCC2)c1.[K+]\",\n        \"reaction_fingerprints\": \"<FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN><FP_TOKEN>\"\n    }\n],\n  \"current_reactants\": \"COC([O-])(O)c1ccccc1-c1ccc(Cl)c(C(=O)NCC2(C)CCCCCC2)c1\"\n}\n\nAnswer:\n{  \n  \"Elementary Step i\": {\n    \"reactants\": \"...\",\n    \"products\": \"...\",\n  }\n} /no_think"}], "gt": "C[O-].CC1(CNC(=O)c2cc(-c3ccccc3C(=O)O)ccc2Cl)CCCCCC1", "task_type": "nepp", "SMILES": [{"idx": "1", "reactants": "O[K].COC(=O)c1ccccc1-c1ccc(Cl)c(C(=O)NCC2(C)CCCCCC2)c1", "products": "COC([O-])(O)c1ccccc1-c1ccc(Cl)c(C(=O)NCC2(C)CCCCCC2)c1.[K+]"}]}
```

### 2. Running Batch Inference

Use the inference.py script to process your .jsonl file. This script supports multi-modal inputs (SMILES fingerprints + Text) and will save the predictions to the output directory.
```
python inference.py \
  --inference_data_path ./test_benchmarks/chemcotbench/nepp-fp.jsonl \
  --model_path ./checkpoints/CoRAL-8B \
  --output_dir ./outputs \
  --temperature 0.3
  --count 1
  --num_beam 1
```

### 3. Arguments Explained

You can adjust the generation parameters to control the diversity and quality of the predictions.

| Argument | Shorthand | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--inference_data_path` | `-i` | `str` | - | **Required.** Path to the input dataset file (must be in `.jsonl` format). |
| `--model_path` | - | `str` | `./checkpoints/CoRAL-8B` | Path to the main LLM checkpoint folder. |
| `--output_dir` | - | `str` | `./outputs` | Directory where the prediction results will be saved. |
| `--temperature` | `-t` | `float` | `0.3` | Controls randomness in generation. Lower values (e.g., 0.1) make the model more deterministic; higher values (e.g., 0.7) make it more creative. |
| `--count` | `-cnt` | `int` | `3` | The number of inference attempts per sample. Useful for generating multiple candidates for complex reactions. |
| `--num_beam` | `-n` | `int` | `1` | Beam search size. Increasing this (e.g., to 3 or 5) explores more paths but slows down inference speed significantly. |

### 4. Output Format

The script will generate a new .jsonl file in the --output_dir. Each line will contain the original input index and the model's generated response:
```
{'response': '<think>\n\n</think>\n\n\n    {\n      "Elementary Step 6": {\n        "reactants": "CCN=C=NCCCN(C)C.O=C(O)c1ccc(Oc2ccc(OCc3ccccc3)cc2)cc1",\n        "products": "CCN=C=[NH+]CCCN(C)C.O=C([O-])c1ccc(Oc2ccc(OCc3ccccc3)cc2)cc1",\n      }\n    }\n    ', 'idx': 0}
```
