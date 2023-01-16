import gc
import logging
from abc import ABC
from typing import Optional

import torch
import wandb
from seqlbtoolkit.base_model.train import BaseTrainer

from ..base.status import Status

logger = logging.getLogger(__name__)


class BaseNERTrainer(BaseTrainer, ABC):
    """
    Bert trainer used for training BERT for token classification (sequence labeling)
    """

    def __init__(self,
                 config,
                 collate_fn=None,
                 model=None,
                 training_dataset=None,
                 valid_dataset=None,
                 test_dataset=None):

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

    def run(self):
        self._model.to(self._device)

        self._status.init(self.config.model_buffer_size, False)
        # ----- start training process -----
        logger.info("Start training...")
        for epoch_i in range(self._config.num_train_epochs):
            logger.info("------")
            logger.info(f"Epoch {epoch_i + 1} of {self._config.num_train_epochs}")

            training_dataloader = self.get_dataloader(self._training_dataset, shuffle=True)

            train_loss = self.training_step(training_dataloader)
            logger.info("Training loss: %.4f" % train_loss)

            self.eval_and_save()

        test_results = self.test()
        for k, v in test_results['micro avg'].items():
            wandb.run.summary[f"test/{k}"] = v
        self.save_results_as_wandb_table(test_results)
        self.save_results_as_csv(test_results)

        logger.info("Test results:")
        self.log_results(test_results)

        wandb.finish()

        gc.collect()
        torch.cuda.empty_cache()

        return None

    def eval_and_save(self):
        """
        Evaluate the model and save it if its performance exceeds the previous highest
        """
        valid_results = self.evaluate(self.valid_dataset)

        logger.info("Validation results:")
        self.log_results(valid_results)

        step_idx = self._status.eval_step + 1

        result_dict = {f"valid/{k}": v for k, v in valid_results.items()}
        wandb.log(data=result_dict, step=step_idx)

        # ----- check model performance and update buffer -----
        if self._status.model_buffer.check_and_update(valid_results.f1, self.model):
            logger.info("Model buffer is updated!")

        self._status.eval_step += 1
        return None

    @staticmethod
    def log_results(metrics):
        if isinstance(metrics, dict):
            for key, val in metrics.items():
                logger.info(f"[{key}]")
                for k, v in val.items():
                    logger.info(f"  {k}: {v:.4f}.")
        else:
            for k, v in metrics.items():
                logger.info(f"  {k}: {v:.4f}.")

    @staticmethod
    def save_results_as_wandb_table(metrics: dict, table_name: Optional[str] = 'test_results'):
        """
        Save dictionary results in a wandb table
        """
        columns = ['Entity Type', 'Precision', 'Recall', 'F1']
        tb = wandb.Table(columns)

        for ent, m in metrics.items():
            row = [ent] + [value for value in m.values()]
            tb.add_data(*row)
        wandb.run.log({table_name: tb})

        return None

    def save_results_as_csv(self, metrics: dict, file_name: Optional[str] = 'test_results.csv'):
        """
        Save dictionary results in a csv file
        """
        import os
        import pandas as pd

        os.makedirs(self.config.output_dir, exist_ok=True)

        columns = ['entity', 'precision', 'recall', 'f1']
        data_dict = {c: list() for c in columns}

        for ent, mtrs in metrics.items():
            data_dict['entity'].append(ent)
            for k, v in mtrs.items():
                data_dict[k].append(v)

        df = pd.DataFrame(data_dict)
        df.to_csv(os.path.join(self.config.output_dir, file_name), index=False)

        return None
