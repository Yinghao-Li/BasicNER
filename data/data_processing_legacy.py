
import os
import sys
import json
import logging
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

from transformers import (
    HfArgumentParser,
    set_seed,
)
from chemdataextractor.doc import Paragraph

from seqlbtoolkit.io import set_logging, logging_args, save_json
from seqlbtoolkit.data import respan, span_dict_to_list, merge_list_of_lists


logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- manage directories and IO ---
    data_dir: Optional[str] = field(
        default='', metadata={'help': 'Directory to datasets'}
    )
    output_dir: Optional[str] = field(
        default='.',
        metadata={"help": "The folder where the models and outputs will be written."},
    )
    log_dir: Optional[str] = field(
        default=None,
        metadata={"help": "the directory of the log file. Set to '' to disable logging"}
    )


def process(args: Arguments):
    ents = list()
    for partition in ('train', 'valid', 'test'):
        with open(os.path.join(args.data_dir, f"{partition}.txt"), 'r', encoding='utf-8') as f:
            instances = json.load(f)

        text_list = [inst['text'] for inst in instances]
        ent_list = [{(d['start_offset'], d['end_offset']): d['label'] for d in instance['entities']}
                    for instance in instances]

        output_dict = dict()
        for idx, (text, ent_spans) in enumerate(zip(text_list, ent_list)):
            raw_tks = Paragraph(' '.join(text)).raw_tokens
            cde_tks = merge_list_of_lists(raw_tks)
            cde_ent_spans = span_dict_to_list(respan(text, cde_tks, ent_spans))
            sent_lengths = [len(tk_seq) for tk_seq in raw_tks]

            output_dict[f"{idx}"] = {"label": cde_ent_spans, "data": {"text": cde_tks, 'sent_lengths': sent_lengths}}

        ents += merge_list_of_lists([[d['label'] for d in instance['entities']] for instance in instances])

        save_json(output_dict, os.path.join(args.output_dir, f"{partition}.json"), collapse_level=4)

    ents = set(ents)
    save_json({'entity_types': list(ents)}, os.path.join(args.output_dir, "meta.json"))


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

    if getattr(arguments, "log_dir", None) is None:
        arguments.log_dir = os.path.join('logs', f'{_current_file_name}', f'{_time}.log')

    set_logging(log_dir=arguments.log_dir)
    logging_args(arguments)

    set_seed(getattr(arguments, 'seed', 42))

    try:
        process(args=arguments)
    except Exception as e:
        logger.exception(e)
        raise e
