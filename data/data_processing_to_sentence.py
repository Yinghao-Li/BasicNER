
import os
import sys
import json
import random
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
from seqlbtoolkit.data import (
    respan,
    span_dict_to_list,
    label_to_span,
    span_to_label,
    merge_list_of_lists,
    split_list_by_lengths
)


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
    ents = set()
    sample_dict = dict()
    for partition in ('train', 'valid', 'test'):
        with open(os.path.join(args.data_dir, f"{partition}.txt"), 'r', encoding='utf-8') as f:
            instances = json.load(f)

        text_list = merge_list_of_lists(instances['text'])
        lbs_list = merge_list_of_lists(instances['label'])

        output_dict = dict()

        idx = 0
        for text, lbs in zip(text_list, lbs_list):
            cde_sents = Paragraph(' '.join(text)).raw_tokens
            cde_tks = merge_list_of_lists(cde_sents)

            ori_spans = label_to_span(lbs)
            ent_spans = {k: v for k, v in ori_spans.items() if v != 'ES'}
            cde_ent_spans = respan(text, cde_tks, ent_spans)
            cde_lbs = span_to_label(cde_ent_spans, cde_tks)

            sent_lengths = [len(tk_seq) for tk_seq in cde_sents]
            cde_sent_lbs = split_list_by_lengths(cde_lbs, sent_lengths)

            cde_sent_ent_spans = [span_dict_to_list(label_to_span(sent_lbs)) for sent_lbs in cde_sent_lbs]

            for sent_tks, sent_ent_spans in zip(cde_sents, cde_sent_ent_spans):

                output_dict[f"{idx}"] = {
                    "label": sent_ent_spans,
                    "data": {"text": sent_tks, "sent_lengths": [len(sent_tks)]}
                }
                idx += 1

            ents.update(ent_spans.values())

        save_json(output_dict, os.path.join(args.output_dir, f"{partition}.json"), collapse_level=4)

        if partition == 'train':
            n_sents = idx

            for n_samples in (5, 10, 20, 50, 100):
                sample_dict[n_samples] = {i: sorted(random.sample(range(n_sents), n_samples)) for i in range(10)}

    save_json(
        {'entity_types': list(ents), 'few_shot': sample_dict},
        os.path.join(args.output_dir, "meta.json"),
        collapse_level=4
    )


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
