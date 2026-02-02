import os
import torch

from swift.llm import get_model_tokenizer, load_dataset, get_template
from swift.utils import get_logger, get_model_parameter_info, seed_everything, find_all_linears
from customized_swift.swift_trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from customized_swift.swift_utils import EncodePreprocessor
from datetime import datetime
from codebook.codebook_train import VQVAE

logger = get_logger()
seed_everything(42)
data_seed = 42

# Hyperparameters for training
model_id_or_path = './Qwen3-8B-FP'  # Qwen3-8B with <FP_TOKEN>
fp_model_path = './codebook/checkpoints/model_1752520982.3927734.pt'  # pretrained fingerprint encoder
dataset = './SFT_dataset/SFT_mechanism_stage2/train-FP-1129-half.jsonl'

time = datetime.now().strftime('%m-%d-%H-%M-%S')
output_dir = f'./exp/{time}'
os.makedirs(output_dir, exist_ok=True)

output_dir = os.path.abspath(os.path.expanduser(output_dir))
logger.info(f'output_dir: {output_dir}')

# hyper-parameters
max_length = 4096
split_dataset_ratio = 0.01  # Split validation set
num_proc = 16  # The number of processes for data loading.


def main():
    # Obtain the model and template, and add a trainable Lora layer on the model.
    model, tokenizer = get_model_tokenizer(model_id_or_path, model_type='qwen3', torch_dtype=torch.bfloat16)
    logger.info(f'model_info: {model.model_info}')
    template = get_template(model.model_meta.template, tokenizer, max_length=max_length)
    template.set_mode('train')

    fp_model = VQVAE(
        input_dim=2048,
        latent_shape=(64, 32),
        num_embeddings=1024,
        embedding_dim=4096,
        commitment_cost=0.25
    )
    fp_model = fp_model.to(device=model.device, dtype=torch.bfloat16)
    fp_model.load_state_dict(torch.load(fp_model_path, map_location="cuda:0"))
    for name, param in fp_model.named_parameters():
        param.requires_grad = False

    # training_args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        weight_decay=0.1,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        logging_first_step=True,
        eval_strategy='steps',
        eval_steps=50000,
        num_train_epochs=0.7,
        metric_for_best_model='loss',
        save_total_limit=1,
        save_strategy='steps',
        save_steps=5000,
        logging_steps=5,
        dataloader_num_workers=1,
        data_seed=data_seed,
        # deepspeed="./ds_config.json"
    )

    total_params = sum(p.numel() for p in model.parameters())
    fp_params = sum(p.numel() for p in fp_model.parameters())
    print(f"fp_model parameters: {fp_params}, llm param: {total_params}")

    # Print model structure and trainable parameters.
    logger.info(f'model: {model}')
    model_parameter_info = get_model_parameter_info(model)
    logger.info(f'model_parameter_info: {model_parameter_info}')

    # Load the dataset, split it into a training set and a validation set,
    # and encode the text data into tokens.
    train_dataset, val_dataset = load_dataset(dataset, split_dataset_ratio=split_dataset_ratio, num_proc=num_proc,
                                              seed=data_seed, remove_unused_columns=False, load_from_cache_file=False)

    logger.info(f'data path: {dataset}')
    logger.info(f'train_dataset: {train_dataset}')
    logger.info(f'val_dataset: {val_dataset}')

    train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
    val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)
    print(f'encoded_train_dataset[0]: {train_dataset[2].keys()}')

    # Get the trainer and start the training.
    model.enable_input_require_grads()  # Compatible with gradient checkpointing
    model.__setattr__("fp_model", fp_model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
    )
    trainer.set_fp_model(model.fp_model)
    trainer.train()


if __name__ == '__main__':
    main()
