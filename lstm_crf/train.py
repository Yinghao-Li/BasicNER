import gc
import logging
from typing import Optional

import torch
import wandb
import numpy as np
from seqlbtoolkit.base_model.eval import Metric
from seqlbtoolkit.base_model.train import BaseTrainer
from tqdm.auto import tqdm

from .status import Status
from .args import Config
from .dataset import Dataset
from .collator import collator
from .model import BiRnnCrf

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
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
            training_dataset=training_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            collate_fn=collate_fn
        )

        self._model = model
        self._optimizer = None
        self._scheduler = None
        self._loss_fn = None
        self._status = Status()
        self.initialize()

    @property
    def training_dataset(self):
        return self._training_dataset

    @property
    def valid_dataset(self):
        return self._valid_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

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

    def run(self):
        self._model.to(self._device)

        self._status.init(self.config.model_buffer_size, False)
        # ----- start training process -----
        logger.info("Start training BERT...")
        for epoch_i in range(self._config.num_train_epochs):
            logger.info("------")
            logger.info(f"Epoch {epoch_i + 1} of {self._config.num_train_epochs}")

            training_dataloader = self.get_dataloader(self._training_dataset, shuffle=True)

            train_loss = self.training_step(training_dataloader)
            logger.info("Training loss: %.4f" % train_loss)

            self.eval_and_save()

        test_results = self.test()
        for k, v in test_results.items():
            wandb.run.summary[f"test/{k}"] = v

        logger.info("Test results:")
        self.log_results(test_results)

        wandb.finish()

        gc.collect()
        torch.cuda.empty_cache()

        return None

    def training_step(self, data_loader):
        """
        Implement each training loop
        """
        train_loss = 0

        self._model.train()
        self._optimizer.zero_grad()

        n_nodes = 0
        for idx, inputs in enumerate(tqdm(data_loader)):
            # get data
            inputs.to(self._device)

            # training step
            logits = self._model(inputs)
            loss = self.compute_loss(logits, inputs.lbs, inputs.padding_mask)
            loss.backward()
            # track loss
            n_batch_node = torch.sum(~inputs.padding_mask).cpu()
            train_loss += loss.detach().cpu() * n_batch_node
            self._optimizer.step()
            self._scheduler.step()
            self._optimizer.zero_grad()

            n_nodes += n_batch_node

            if self.config.eval_interval and (idx + 1) % self.config.eval_interval == 0:
                self.eval_and_save()
                self._model.train()

        train_loss /= n_nodes

        return train_loss

    def eval_and_save(self):
        """
        Evaluate the model and save it if its performance exceeds the previous highest
        """
        valid_results = self.evaluate(self.valid_dataset)

        logger.info("Validation results:")
        self.log_results(valid_results)

        step_idx = self._status.eval_step + 1

        result_dict = {
            "valid/precision": valid_results.precision,
            "valid/recall": valid_results.recall,
            "valid/f1": valid_results.f1
        }
        wandb.log(data=result_dict, step=step_idx)

        # ----- check model performance and update buffer -----
        if self._status.model_buffer.check_and_update(valid_results.f1, self.model):
            logger.info("Model buffer is updated!")

        self._status.eval_step += 1
        return None

    def compute_loss(self,
                     logits: torch.Tensor,
                     lbs: torch.Tensor,
                     mask: torch.Tensor):
        """
        Compute training loss
        """
        mask = (~mask).view(-1)

        logits = logits.view(-1)[mask]
        lbs = lbs.view(-1)[mask]

        loss = self._loss_fn(input=logits, target=lbs)
        return loss

    def evaluate(self, dataset: Dataset):
        data_loader = self.get_dataloader(dataset)
        self._model.to(self._device)
        self._model.eval()

        pred_lbs = list()
        true_lbs = list()
        with torch.no_grad():
            for inputs in data_loader:
                inputs.to(self._device)
                lbs = inputs.lbs

                logits = self._model(inputs)

                mask = ~inputs.padding_mask.view(-1)
                pred_prob_batch = torch.sigmoid(logits.view(-1)[mask]).detach().cpu().numpy()
                pred_lb_batch = (pred_prob_batch > 0.5).astype(int)
                pred_lbs.append(pred_lb_batch)

                true_lb_batch = lbs.view(-1)[mask].cpu().numpy().astype(int)
                true_lbs.append(true_lb_batch)

        true_lbs = np.concatenate(true_lbs)
        pred_lbs = np.concatenate(pred_lbs)

        # precision, recall, f1 = binary_precision_recall_f1(true_lbs, pred_lbs)
        # metric = Metric(precision, recall, f1)
        #
        # return metric
        return Metric()

    def predict(self, dataset: Dataset):
        data_loader = self.get_dataloader(dataset)
        self._model.to(self._device)
        self._model.eval()

        pred_probs = list()
        pred_lbs = list()
        with torch.no_grad():
            for inputs in data_loader:
                inputs.to(self._device)

                logits = self._model(inputs)

                mask = ~inputs.padding_mask.view(-1)
                pred_prob_batch = torch.sigmoid(logits.view(-1)[mask]).detach().cpu().numpy()
                pred_probs.append(pred_prob_batch)
                pred_lb_batch = (pred_prob_batch > 0.5).astype(int)
                pred_lbs.append(pred_lb_batch)

        pred_probs = np.concatenate(pred_probs)
        pred_lbs = np.concatenate(pred_lbs)

        return pred_lbs, pred_probs

    def test(self):

        if self._status.model_buffer.size == 1:
            self._model.load_state_dict(self._status.model_buffer.model_state_dicts[0])
            return super().test()

        pred_lbs_list = list()
        for model_state_dicts in self._status.model_buffer.model_state_dicts:
            self._model.load_state_dict(model_state_dicts)
            pred_lbs = self.predict(self.test_dataset)[0]
            pred_lbs_list.append(pred_lbs)

        pred_lbs = np.stack(pred_lbs_list, axis=0).mean(axis=0)
        pred_lbs = (pred_lbs > 0.5).astype(int)

        true_lbs = torch.cat(self.test_dataset.lbs).cpu().numpy()

        # precision, recall, f1 = binary_precision_recall_f1(true_lbs, pred_lbs)
        # metric = Metric(precision, recall, f1)
        #
        # return metric
        return Metric()

    @staticmethod
    def log_results(metrics):
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}.")
