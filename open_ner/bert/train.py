import logging
from typing import Optional

import torch
import numpy as np
from seqlbtoolkit.base_model.eval import get_ner_metrics
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_scheduler

from .args import Config
from .dataset import Dataset
from .collator import DataCollator
from ..base.train import BaseNERTrainer

logger = logging.getLogger(__name__)


class Trainer(BaseNERTrainer):
    """
    Bert trainer used for training BERT for token classification (sequence labeling)
    """

    def __init__(self,
                 config: Config,
                 collate_fn=None,
                 model=None,
                 training_dataset: Optional[Dataset] = None,
                 valid_dataset: Optional[Dataset] = None,
                 test_dataset: Optional[Dataset] = None):

        if not collate_fn:
            tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name_or_path)
            collate_fn = DataCollator(tokenizer)

        super().__init__(
            config=config,
            collate_fn=collate_fn,
            model=model,
            training_dataset=training_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
        )

    def initialize_model(self):
        self._model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self._config.bert_model_name_or_path,
            num_labels=self.config.n_lbs
        )
        return self

    def initialize_optimizer(self, optimizer=None):
        """
        Initialize training optimizer
        """
        if optimizer is not None:
            self._optimizer = optimizer
        else:
            self._optimizer = torch.optim.AdamW(
                self._model.parameters(),
                lr=self._config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        return self

    def initialize_scheduler(self, scheduler=None):
        """
        Initialize learning rate scheduler
        """
        num_update_steps_per_epoch = int(np.ceil(
            len(self._training_dataset) / self._config.batch_size
        ))
        num_warmup_steps = int(np.ceil(
            num_update_steps_per_epoch * self._config.warmup_ratio * self._config.num_train_epochs
        ))
        num_training_steps = int(np.ceil(num_update_steps_per_epoch * self._config.num_train_epochs))

        self._scheduler = get_scheduler(
            self._config.lr_scheduler_type,
            self._optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return self

    def training_step(self, data_loader):
        """
        Implement each training loop
        """
        train_loss = 0

        self._model.train()
        self._optimizer.zero_grad()

        for idx, inputs in enumerate(tqdm(data_loader)):
            # get data
            inputs.to(self._device)

            # training step
            loss = self._model(**inputs.__dict__).loss
            loss.backward()
            # track loss
            train_loss += loss.detach().cpu() * len(inputs)
            self._optimizer.step()
            self._scheduler.step()
            self._optimizer.zero_grad()

        return train_loss

    def evaluate(self, dataset: Dataset, detailed_result: Optional[bool] = False):
        data_loader = self.get_dataloader(dataset)
        self._model.to(self._device)
        self._model.eval()

        pred_lbs = list()
        with torch.no_grad():
            for inputs in data_loader:
                inputs.to(self._device)

                logits = self._model(**inputs.__dict__).logits
                pred_ids = logits.argmax(-1).detach().cpu()

                pred_lb_batch = [[self.config.bio_label_types[i] for i in pred[lbs >= 0]]
                                 for lbs, pred in zip(inputs.labels.cpu(), pred_ids)]
                pred_lbs += pred_lb_batch

        metric = get_ner_metrics(dataset.lbs, pred_lbs, detailed=detailed_result)
        return metric

    def test(self):

        if self._status.model_buffer.size == 1:
            self._model.load_state_dict(self._status.model_buffer.model_state_dicts[0])
            metrics = self.evaluate(self._test_dataset, detailed_result=True)
            return metrics

        raise NotImplementedError("Function for multi-checkpoint caching & evaluation is not implemented!")
