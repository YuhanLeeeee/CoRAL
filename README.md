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
python demo_step_pred.py --input "YOUR_SMILES_STRING_HERE"
# Or simply run the default script:
python demo_step_pred.py
```

**Expected Output:**

Upon running the script, the model will load the weights and output the predicted elementary step product with its confidence score.

**Expected Run Time:**

GPU (e.g., A800): < 10 seconds.

### Case 2: Reaction Prediction (RP)

This demonstration predicts the final products of chemical reaction sequences, as well as the overall elementary reaction pathways.

**Run the demo:**
```
python demo_reaction_pred.py --input "YOUR_Reactants_SMILES"
# Or run with default examples:
python demo_reaction_pred.py
```

**Expected Output:**

The script will output the predicted final product SMILES string.

**Expected Run Time:**

GPU (e.g., A800): < 40 seconds.
