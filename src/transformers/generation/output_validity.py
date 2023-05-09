from typing import Optional, Iterable, Tuple, List, Union, Dict, Any
import torch

from scs.incremental_parse import SpecialToken
from scs.handler import JSONSchemaCheckFactory, JSONValidityCheckFactory, SyntaxValidityCheckHandler, OneOfValidityCheckFactory

from ..tokenization_utils import PreTrainedTokenizer


SP_START_TOKEN = b'\xe2\x96\x81'.decode('utf-8')


def get_token_vocab(tokenizer: PreTrainedTokenizer) -> List[Union[str, SpecialToken]]:
    vocab_list = tokenizer.batch_decode(torch.arange(tokenizer.vocab_size, dtype=torch.long)[:,None])

    # TODO need better solution for detecting sentencepiece
    if type(tokenizer).__name__ in ["LlamaTokenizer", "LlamaTokenizerFast"]:
        is_start_token = [tokenizer.convert_ids_to_tokens(i).startswith(SP_START_TOKEN) for i in range(tokenizer.vocab_size)]
        vocab_list = [(' ' if is_start else '') + t for t, is_start in zip(vocab_list, is_start_token)]

    vocab_list[tokenizer.eos_token_id] = [SpecialToken.EOS]
    return vocab_list


def validity_check(tokenizer: PreTrainedTokenizer, kwargs: Dict[str, Any]) -> Optional[SyntaxValidityCheckHandler]:
    num_workers = kwargs.pop("constraint_workers", 1)
    json_validity_check_factory = JSONValidityCheckFactory(
        allow_outer_list=kwargs.pop("allow_outer_list", True),
        allow_empty=kwargs.pop("allow_empty", True),
        allow_empty_children=kwargs.pop("allow_empty_children", True),
        allow_whitespace_formatting=kwargs.pop("allow_whitespace_formatting", False),
    )
    one_of_match_strings = [s for s in kwargs.pop("enforce_one_of", "").split(",") if s]
    enforce_json = kwargs.pop("enforce_json", False)
    enforce_json_schema = kwargs.pop("enforce_json_schema", None)
    if enforce_json_schema:
        return SyntaxValidityCheckHandler(
            get_token_vocab(tokenizer),
            JSONSchemaCheckFactory(schema=enforce_json_schema),
            num_workers=num_workers,
        )
    if enforce_json:
        return SyntaxValidityCheckHandler(
            get_token_vocab(tokenizer),
            json_validity_check_factory,
        )
    if one_of_match_strings:
        return SyntaxValidityCheckHandler(
            get_token_vocab(tokenizer),
            OneOfValidityCheckFactory(match_strings=one_of_match_strings),
        )
