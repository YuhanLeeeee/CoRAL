#!/bin/bash
#srun --gres=gpu:a800:8 -p ai main.sh
#sbatch --gres=gpu:a800:8 -p ai main.sh
module load CUDA/11.8
module load Pytorch/2.7.0-py3.9_cuda11.8
module load nccl/2.16.5-cuda-11.8

module list
nvidia-smi

echo `which python`

python SFT.py
# deepspeed --num_gpus 8 SFT.py
#python GRPO.py

#python inference_example.py

#python reference.py -i ./test_benchmarks/in-distribution/USPTO_test_nepp.jsonl --temperature 0.3 -cnt 1
#python reference.py -i ./test_benchmarks/in-distribution/USPTO_test_fs.jsonl --temperature 0.3 -cnt 1 -n 5
#python reference.py -i ./test_benchmarks/chemcotbench/reaction/nepp-fp.jsonl --temperature 0.3 -cnt 1
#python reference.py -i ./test_benchmarks/chemcotbench/reaction/fs-CO.jsonl --temperature 0.3 -cnt 3
#python reference.py -i ./test_benchmarks/mol-instr/reaction/Mol-Instr-fs.jsonl --temperature 0.3 -cnt 3
#python reference.py -i ./test_benchmarks/perturbation/fs-perturbation-300.jsonl --temperature 0.3 -cnt 3
#python reference.py -i ./test_benchmarks/prospective_test/rxn4000.jsonl
# Yield Prediction
#python ./reference_yield.py -rt SM_test11 --cnt 6 --total 6
#python ./yield_prediction/DRFP_MLP.py
