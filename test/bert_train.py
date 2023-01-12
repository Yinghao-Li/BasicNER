# coding=utf-8

import logging
import os
import sys
import wandb
from datetime import datetime

from transformers import (
    HfArgumentParser,
    set_seed,
)

from seqlbtoolkit.io import set_logging, logging_args

from open_ner.bert.args import Arguments, Config
from open_ner.bert.dataset import Dataset
from open_ner.bert.train import Trainer

logger = logging.getLogger(__name__)


def main(args):

    config = Config().from_args(args).get_meta()

    if args.apply_wandb and args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=config.__dict__,
        mode='online' if args.apply_wandb else 'disabled'
    )

    # ------------
    # data
    # ------------

    logger.info('Loading datasets...')
    training_dataset = Dataset().prepare(
        config=config,
        partition='train'
    )
    logger.info(f'Training dataset loaded, length={len(training_dataset)}')

    valid_dataset = Dataset().prepare(
        config=config,
        partition='valid'
    )
    logger.info(f'Validation dataset loaded, length={len(valid_dataset)}')

    test_dataset = Dataset().prepare(
        config=config,
        partition='test'
    )
    logger.info(f'Test dataset loaded, length={len(test_dataset)}')

    # ------------
    # training
    # ------------

    treeformer_trainer = Trainer(
        config=config,
        training_dataset=training_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
    ).initialize()

    treeformer_trainer.run()

    return None


if __name__ == '__main__':

    _time = datetime.now().strftime("%m.%d.%y-%H.%M")
    _current_file_name = os.path.basename(__file__)
    if _current_file_name.endswith('.py'):
        _current_file_name = _current_file_name[:-3]

    # --- set up arguments ---
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        arguments, = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        arguments, = parser.parse_args_into_dataclasses()

    if not getattr(arguments, "log_dir", None):
        arguments.log_dir = os.path.join('../logs', f'{_current_file_name}', f'{_time}.log')

    set_logging(log_dir=arguments.log_dir)
    logging_args(arguments)

    set_seed(arguments.seed)

    main(args=arguments)
