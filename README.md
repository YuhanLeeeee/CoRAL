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
