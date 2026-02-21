# CoRAL: Physically Grounded Multi-modal Reasoning for Organic Reaction Mechanism Prediction

This is the official repository for the paper **"Physically Grounded Multi-modal Reasoning for Organic Reaction Mechanism Prediction"**.

> ⚠️ **Note**
>
> **This repository is continuously being updated. We are actively refining the documentation and uploading more checkpoints and experimental data.**

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

## ⚙️ System Requirements (系统要求)

### 1. Hardware Requirements (硬件要求)
<!-- 对应图片第3点：非标准硬件要求 -->
This project involves large-scale model training/inference and requires high-performance computing resources.

*   **Recommended Hardware:** 
    *   **GPU:** At least one NVIDIA GPU with **80GB VRAM** (or higher).
    *   **Architecture:** Ampere (A100/A800) or Hopper (H100/H800) recommended.
*   **Tested Hardware:** 
    *   **GPU:** NVIDIA A800 80GB PCIe
    *   **Memory:** 80GB Dedicated Video Memory

### 2. Software & Key Dependencies (软件依赖)
<!-- 对应图片第1点：所有依赖和版本号 -->
The project relies on specific versions of deep learning frameworks. Below are the core dependencies:

*   **Python Version:** Python 3.10+ (Recommended)
*   **Core Libraries:**
    *   `ms_swift` == **3.7.3**
    *   `torch` == **2.7.0+cu118** (CUDA 11.8 build)
    *   `transformers` == **4.52.4**
    *   `deepspeed` == **0.17.2**
    *   `unsloth` == **2026.2.1**
    *   `vllm_client` == **0.3.2.0**

> **Note:** For a complete list of all dependencies, please refer to [requirements.txt](./requirements.txt).

### 3. Tested Environment (测试环境)
<!-- 对应图片第2点：已测试的版本和环境 -->
The software has been rigorously tested and verified on the following environment setup:

| Component | Version / Details |
| :--- | :--- |
| **OS** | Linux (Ubuntu 20.04/22.04) |
| **NVIDIA Driver** | **535.104.12** |
| **CUDA Version** | **12.2** (System-level) |
| **GPU** | NVIDIA A800 80GB PCIe |

---

## 🚀 Installation (安装指南)

To set up the environment with all the required dependencies, please follow these steps:

1.  **Create a virtual environment (Recommended):**
    ```bash
    conda create -n my_env python=3.10
    conda activate my_env
    ```

2.  **Install dependencies:**
    Run the following command to install all packages listed in `requirements.txt`.
    
    ```bash
    pip install -r requirements.txt
    ```

    *Note: Since `torch` specifies a local version identifier (`+cu118`), ensure your pip allows version specifiers or use the extra index URL if necessary:*
    ```bash
    pip install torch==2.7.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    ```
