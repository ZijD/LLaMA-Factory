import os
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Tuple

import torch
from torch import nn
from transformers import DataCollatorForLanguageModeling
from transformers.trainer_utils import EvalPrediction
from transformers.modeling_utils import PreTrainedModel

from ..data import get_template_and_fix_tokenizer, get_dataset
from ..extras import logging
from ..extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..hparams import get_infer_args, get_train_args
from ..model import load_model, load_tokenizer
from ..train.pt.trainer import CustomTrainer

from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM


if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = logging.get_logger(__name__)

class EmbedPersistent:
    def __init__(self):
        self.embed_list = []
        
    def compute_embeds(self, obj: EvalPrediction, compute_result: bool = False):
        self.embed_list.append(torch.mean(obj.predictions[1][-1], dim=1))
        return {}



class Embedder(CustomTrainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        ret = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        self.embed_list = []
        return ret
    

    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.

            Subclass and override for custom behavior.
            """
            if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            if self.model_accepts_loss_kwargs:
                loss_kwargs = {}
                if num_items_in_batch is not None:
                    loss_kwargs["num_items_in_batch"] = num_items_in_batch
                inputs = {**inputs, **loss_kwargs}
            outputs = model.forward(output_hidden_states=True, **inputs)
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            return (loss, outputs) if return_outputs else loss

def get_embed(args: Optional[Dict[str, Any]] = None):
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="pt", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    embed_store = EmbedPersistent()
    embedder = Embedder(
        model=model,
        compute_metrics=embed_store.compute_embeds,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        **dataset_module,
        **tokenizer_module,
    )
    ret = embedder.evaluate(metric_key_prefix='embed')
    embeds = torch.concat(embed_store.embed_list)
    print(embeds.shape)
    with open(os.path.join(training_args.output_dir, 'embed.pth'), 'wb') as f:
        torch.save(embeds.cpu(), f)
        f.close()