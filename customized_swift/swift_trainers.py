# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import inspect
import os
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

from torch_geometric.data import Batch, Data
import einops
import torch
from peft import PeftModel
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import EvalPrediction
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available

from swift.utils import JsonlWriter, Serializer, gc_collect, get_logger
from swift.trainers.arguments import Seq2SeqTrainingArguments, TrainingArguments
from .swift_mixin import DataLoaderMixin, SwiftMixin

logger = get_logger()


class Trainer(SwiftMixin, HfTrainer):
    args: TrainingArguments

    @contextmanager
    def _patch_loss_function(self):
        model = self.model
        if isinstance(model, PeftModel):
            model = model.model
        model_cls = model.__class__
        if not hasattr(model_cls, 'loss_function'):
            yield
            return

        loss_function = model.loss_function
        _old_loss_function = model_cls.loss_function

        @staticmethod
        @wraps(loss_function)
        def new_loss_function(logits, labels, **kwargs):
            labels = labels.to(logits.device)  # fix device_map
            return loss_function(logits=logits, labels=labels, **kwargs)

        model_cls.loss_function = new_loss_function
        try:
            yield
        finally:
            model_cls.loss_function = _old_loss_function

    def train(self, *args, **kwargs):
        with self._patch_loss_function():
            return super().train(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if inputs.get('labels') is not None:
            self._compute_acc(outputs, inputs['labels'])
        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss /= self.args.gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss


class EmbeddingTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = self.calculate_metric
        self.preprocess_logits_for_metrics = None
        self.label_names = ['labels']

    def calculate_metric(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        from swift.plugin.loss import infonce_loss, calculate_paired_metrics, calculate_infonce_metrics
        if self.compute_loss_func is infonce_loss:
            return calculate_infonce_metrics(eval_prediction.predictions, eval_prediction.label_ids)
        else:
            return calculate_paired_metrics(eval_prediction.predictions, eval_prediction.label_ids)


class Seq2SeqTrainer(SwiftMixin, DataLoaderMixin, HfSeq2SeqTrainer):
    args: Seq2SeqTrainingArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True  # fix transformers>=4.46.2
        if self.args.predict_with_generate:
            from swift.llm import PtEngine
            self.infer_engine = PtEngine.from_model_template(
                self.model, self.template, max_batch_size=self.args.per_device_eval_batch_size)
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'predict.jsonl'))

    def set_fp_model(self, fp_model):
        self.fp_model = fp_model

    def set_graph_model(self, graph_model):
        self.graph_model = graph_model

    @staticmethod
    def _predict_data_collator(batch):
        return {'_data': batch}

    @contextmanager
    def _patch_predict_with_generate(self):
        origin_mode = self.template.mode
        self.template.set_mode('pt')
        is_multimodal = self.model.model_meta.is_multimodal
        origin_data_collator = self.data_collator

        if is_multimodal:
            models = self.template.remove_post_encode_hook()
        self.data_collator = self._predict_data_collator
        try:
            yield
        finally:
            if is_multimodal:
                self.template.register_post_encode_hook(models)
            self.data_collator = origin_data_collator
            self.template.set_mode(origin_mode)

    def evaluate(self, *args, **kwargs):
        context = self._patch_predict_with_generate() if self.args.predict_with_generate else nullcontext()
        with context:
            res = super().evaluate(*args, **kwargs)
            gc_collect()
            return res

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
            **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
        from swift.llm import RequestConfig, InferRequest
        data_list = inputs['_data']
        labels_list = [InferRequest.remove_response(data['messages']) for data in data_list]
        resp_list = self.infer_engine.infer(
            data_list,
            RequestConfig(max_tokens=self.model.generation_config.max_new_tokens),
            use_tqdm=False,
            template=self.template)

        response_list = []
        jsonl_cache = []
        device = self.args.device
        for data, resp, labels in zip(data_list, resp_list, labels_list):
            response = resp.choices[0].message.content
            jsonl_cache.append({'response': response, 'labels': labels, **data})
            response_list.append(Serializer.to_tensor(resp.choices[0].message.content).to(device=device))
        self.jsonl_writer.append(jsonl_cache, gather_obj=True)
        labels_list = [Serializer.to_tensor(labels).to(device=device) for labels in labels_list]
        response_list = pad_sequence(response_list, batch_first=True, padding_value=0)
        labels_list = pad_sequence(labels_list, batch_first=True, padding_value=0)
        return None, response_list, labels_list

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # allocated_memory = torch.cuda.memory_allocated('cuda:0') / (1024 ** 3)  # GB
        # print(f"Allocated Memory (Tensors): {allocated_memory:.2f} GB")

        from swift.plugin.loss import get_loss_func
        loss_kwargs = {}
        labels = None
        compute_loss_func = self.compute_loss_func
        loss_scale = inputs.pop('loss_scale', None)
        if loss_scale is not None:
            loss_kwargs['loss_scale'] = loss_scale
            if compute_loss_func is None:
                compute_loss_func = get_loss_func('loss_scale')

        sample_channels = inputs.pop('channel', None)
        if sample_channels is not None and self.args.channels is not None:
            state = self.state
            setattr(state, 'local_step', getattr(state, 'local_step', 0))
            setattr(state, 'ch_loss_steps', getattr(state, 'ch_loss_steps', {}))

            loss_kwargs['sample_channels'] = sample_channels
            loss_kwargs['trainer'] = self

        if (self.label_smoother is not None or compute_loss_func is not None) and 'labels' in inputs:
            labels = inputs.pop('labels')

        if 'CausalLM' in str(type(model)):
            embedding_layer = model.get_input_embeddings()
        else:
            embedding_layer = model.module.get_input_embeddings()

        # insert fingerprint embedding
        input_ids = inputs["input_ids"]
        batch_size = input_ids.shape[0]
        embeddings = embedding_layer(input_ids)
        # TODO
        ''' 
        You must modify the file ~/.conda/envs/YOUR_ENV/lib/python3.10/site-packages/swift/llm/template/base.py 
        and add the line "if 'fp' in batch[0].keys(): res['fp'] = self.gather_list(batch, 'fp')" at line 1607 
        to pass this attribute value!
        '''
        if 'fp' in inputs.keys() and len(inputs['fp']) != 0:
            batch_indices, token_indices = torch.where(input_ids == 151669)
            fp_token_size = 64
            fingerprints = torch.tensor(inputs['fp'], dtype=torch.bfloat16).to(embeddings.device)
            if len(fingerprints.shape) == 2:
                fingerprints = fingerprints.unsqueeze(dim=0)
            fp_embeddings = self.fp_model.quantize(fingerprints).to(torch.bfloat16)
            fp_embeddings = torch.reshape(fp_embeddings,
                                          (batch_size, int(len(batch_indices) / fp_token_size), fp_token_size, 4096))
            mask = torch.zeros_like(embeddings, dtype=torch.bool)
            mask[batch_indices, token_indices] = True
            new_embeddings = embeddings.clone().to(torch.bfloat16)
            new_embeddings = new_embeddings.masked_scatter(mask, fp_embeddings)
            embeddings = new_embeddings
        outputs = model(
            inputs_embeds=embeddings,
            attention_mask=inputs.get("attention_mask", None),
            labels=inputs.get("labels", None),
            token_type_ids=inputs.get("token_type_ids", None),
            logits_to_keep=inputs.get("logits_to_keep", None)
        )
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is None:
            labels = inputs['labels']
            outputs.loss = outputs.loss.to(labels.device)
            # fix https://github.com/huggingface/transformers/issues/34263
            if num_items_in_batch is not None:
                # print("before:", outputs.loss)
                outputs.loss = outputs.loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)
                # print("after:", outputs.loss)

            if isinstance(outputs, dict) and 'loss' not in outputs:
                raise ValueError(
                    'The model did not return a loss from the inputs, only the following keys: '
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        else:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if compute_loss_func is not None:
                loss = compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, **loss_kwargs)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)

        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            loss = sequence_parallel.reduce_outputs(loss, labels)

        if getattr(self.args, 'average_tokens_across_devices', False) and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        if outputs.logits is not None and labels is not None and not return_outputs:
            # Liger does not have logits
            self._compute_acc(outputs, labels)
        # logger.log("loss")
        return (loss, outputs) if return_outputs else loss
