import logging
from typing import Optional

import torch
from seqlbtoolkit.base_model.eval import get_ner_metrics
from tqdm.auto import tqdm

from .args import Config
from .dataset import Dataset
from .collator import collator
from .model import BiRnnCrf
from ..base.train import BaseNERTrainer

logger = logging.getLogger(__name__)


class Trainer(BaseNERTrainer):
    """
    Bert trainer used for training BERT for token classification (sequence labeling)
    """

    def __init__(self,
                 config: Config,
                 collate_fn=collator,
                 model=None,
                 training_dataset: Optional[Dataset] = None,
                 valid_dataset: Optional[Dataset] = None,
                 test_dataset: Optional[Dataset] = None):

        super().__init__(
            config=config,
            collate_fn=collate_fn,
            model=model,
            training_dataset=training_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
        )

    def initialize_model(self):
        self._model = BiRnnCrf(config=self.config)
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
            loss = self._model.loss(inputs)
            loss.backward()
            # track loss
            train_loss += loss.detach().cpu()
            self._optimizer.step()
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

                _, pred_ids = self._model(inputs)

                pred_lb_batch = [[self.config.bio_label_types[lb_index] for lb_index in label_indices]
                                 for label_indices in pred_ids]
                pred_lbs += pred_lb_batch

        true_lbs = [[self.config.bio_label_types[lb_index] for lb_index in label_indices]
                    for label_indices in dataset.lbs]
        metric = get_ner_metrics(true_lbs, pred_lbs, detailed=detailed_result)
        return metric

    def test(self):

        if self._status.model_buffer.size == 1:
            self._model.load_state_dict(self._status.model_buffer.model_state_dicts[0])
            metrics = self.evaluate(self._test_dataset, detailed_result=True)
            return metrics

        raise NotImplementedError("Function for multi-checkpoint caching & evaluation is not implemented!")
