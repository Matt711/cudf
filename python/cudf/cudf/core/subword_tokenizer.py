# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from __future__ import annotations

import warnings

import cupy as cp

from pylibcudf.nvtext.subword_tokenize import (
    Hashed_Vocabulary as cpp_hashed_vocabulary,
)

from cudf._lib.nvtext.subword_tokenize import (
    subword_tokenize_inmem_hash as cpp_subword_tokenize,
)


def _cast_to_appropriate_type(ar, cast_type):
    if cast_type == "cp":
        return ar

    if cast_type == "pt":
        from torch.utils.dlpack import from_dlpack

    elif cast_type == "tf":
        from tensorflow.experimental.dlpack import from_dlpack

    return from_dlpack(ar.astype("int32").toDlpack())


class SubwordTokenizer:
    """
    Run CUDA BERT subword tokenizer on cuDF strings column.
    Encodes words to token ids using vocabulary from a pretrained
    tokenizer.
    This function requires about 21x the number of character bytes
    in the input strings column as working memory.

    Parameters
    ----------
    hash_file : str
        Path to hash file containing vocabulary of words with token-ids.
        This can be created from the raw vocabulary
        using the ``cudf.utils.hash_vocab_utils.hash_vocab`` function

    do_lower : bool, Default is True
        If set to True, original text will be lowercased before encoding.

    Returns
    -------
    SubwordTokenizer
    """

    def __init__(self, hash_file: str, do_lower_case: bool = True):
        self.do_lower_case = do_lower_case
        self.vocab_file = cpp_hashed_vocabulary(hash_file)

    def __call__(
        self,
        text,
        max_length: int,
        max_num_rows: int,
        add_special_tokens: bool = True,
        padding: str = "max_length",
        truncation: bool | str = False,
        stride: int = 0,
        return_tensors: str = "cp",
        return_token_type_ids: bool = False,
    ):
        """
        Run CUDA BERT subword tokenizer on cuDF strings column.
        Encodes words to token ids using vocabulary from a
        pretrained tokenizer.

        Parameters
        ----------
        text : cudf string series
            The batch of sequences to be encoded.

        max_length : int
            Controls the maximum length to use or pad to.

        max_num_rows : int
            Maximum number of rows for the output token-ids expected to
            be generated by the tokenizer.
            Used for allocating temporary working memory on the GPU device.
            If the output generates a larger number of rows,
            behavior is undefined.
            This will vary based on stride, truncation, and max_length.
            For example, for non-overlapping sequences output rows will be
            the same as input rows.
            A good default can be twice the max_length

        add_special_tokens : bool, optional, defaults to True
            Whether or not to encode the sequences with the special tokens
            of the BERT classification model

        padding : "max_length"
            Pad to a maximum length specified with the argument max_length

        truncation : bool, defaults to False
            True:
            Truncate to a maximum length specified with the argument max_length
            False or 'do_not_truncate': default
            No truncation (Output differs from HuggingFace)

        stride : int, optional, defaults to 0
            The value of this argument defines the number of
            overlapping tokens.
            The information about the overlapping tokens is
            present in the metadata outputted.

        return_tensors : str, {"cp", "pt", "tf"} defaults to "cp"
            "cp" : Return cupy cp.ndarray objects
            "tf" : Return TensorFlow tf.constant objects
            "pt" : Return PyTorch torch.Tensor objects


        return_token_type_ids : bool, optional
            Only False currently supported

        Returns
        -------
        An encoding with the following fields:
            input_ids:(type defined by return_tensors)
                A tensor of token ids to be fed to the model.
            attention_mask: (type defined by return_tensors)
                A tensor of indices specifying which tokens
                should be attended to by the model
            metadata: (type defined by return_tensors)
                Each row contains the index id of the original string and the
                first and last index of the token-ids that are non-padded and
                non-overlapping

        Examples
        --------
        >>> import cudf
        >>> from cudf.utils.hash_vocab_utils import hash_vocab
        >>> hash_vocab('bert-base-cased-vocab.txt', 'voc_hash.txt')


        >>> from cudf.core.subword_tokenizer import SubwordTokenizer
        >>> cudf_tokenizer = SubwordTokenizer('voc_hash.txt',
        ...                                    do_lower_case=True)
        >>> str_series = cudf.Series(['This is the', 'best book'])
        >>> tokenizer_output = cudf_tokenizer(str_series,
        ...                                   max_length=8,
        ...                                   max_num_rows=len(str_series),
        ...                                   padding='max_length',
        ...                                   return_tensors='pt',
        ...                                   truncation=True)
        >>> tokenizer_output['input_ids']
        tensor([[ 101, 1142, 1110, 1103,  102,    0,    0,    0],
                [ 101, 1436, 1520,  102,    0,    0,    0,    0]],
                device='cuda:0',
               dtype=torch.int32)
        >>> tokenizer_output['attention_mask']
        tensor([[1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0]],
                device='cuda:0', dtype=torch.int32)
        >>> tokenizer_output['metadata']
        tensor([[0, 1, 3],
                [1, 1, 2]], device='cuda:0', dtype=torch.int32)
        """

        if return_token_type_ids:
            # raise not currently supported
            # Can also return zeros
            error_msg = "Returning token_type_ids is currently supported"
            raise NotImplementedError(error_msg)

        if truncation in (False, "do_not_truncate"):
            if add_special_tokens:
                error_msg = (
                    "Adding special tokens is not supported "
                    f"with truncation = {truncation}. "
                )
                recommendation = (
                    "Custom Cupy kernel can potentially "
                    "be used to add it. For reference "
                    "see: _bert_add_special_tokens"
                )
                raise NotImplementedError(error_msg + recommendation)

            truncation = False
            warning_msg = (
                "When truncation is not True, the behavior currently differs "
                "from HuggingFace as cudf always returns overflowing tokens"
            )
            warnings.warn(warning_msg)

        if padding != "max_length":
            error_msg = (
                "Only padding to the provided max_length"
                "is currently supported"
            )
            raise NotImplementedError(error_msg)

        if max_length <= stride:
            error_msg = "Stride should be less than max_length"
            raise ValueError(error_msg)

        if return_tensors not in {"cp", "pt", "tf"}:
            error_msg = (
                "Only cupy(cp), pytorch(pt) and tensorflow(tf) "
                "tensors are supported"
            )
            raise NotImplementedError(error_msg)

        stride = max_length - stride
        # behavior varies from subword_tokenize but maps with huggingface

        input_ids, attention_mask, metadata = cpp_subword_tokenize(
            text._column,
            self.vocab_file,
            max_sequence_length=max_length,
            stride=stride,
            do_lower=self.do_lower_case,
            do_truncate=truncation,
        )

        tokenizer_output = {
            "input_ids": cp.asarray(input_ids).reshape(-1, max_length),
            "attention_mask": cp.asarray(attention_mask).reshape(
                -1, max_length
            ),
            "metadata": cp.asarray(metadata).reshape(-1, 3),
        }

        if add_special_tokens:
            tokenizer_output = _bert_add_special_tokens(tokenizer_output)

        tokenizer_output = {
            k: _cast_to_appropriate_type(v, return_tensors)
            for k, v in tokenizer_output.items()
        }

        return tokenizer_output


def _bert_add_special_tokens(token_o):
    """
    Adds special tokens (CLS,SEP) which are often used by pre-trained BERT
    models to input_ids and adjusts attention_mask and metadata to account
    for them.
    """
    max_length = token_o["input_ids"].shape[1]
    seq_end_col = max_length - (token_o["input_ids"][:, ::-1] != 0).argmax(1)
    # clipping to take overflow into account
    seq_end_col = cp.clip(seq_end_col + 1, a_min=None, a_max=max_length - 1)

    _bert_add_special_tokens_input_ids(token_o["input_ids"], seq_end_col)
    _bert_add_special_tokens_attention_mask(
        token_o["attention_mask"], seq_end_col
    )
    _bert_add_special_tokens_metadata(token_o["metadata"], max_length)

    return token_o


def _bert_add_special_tokens_input_ids(input_ids, seq_end_col):
    """
    Add token ids for special tokens ([CLS] and [SEP]) to
    the start and end of each sequence
    """
    # Mark sequence start with [CLS] token mapping to the start of sequence
    input_ids[:, 1:-1] = input_ids[:, 0:-2]
    input_ids[:, 0] = 101
    # Mark end of sequence [SEP]

    input_ids[
        cp.arange(0, input_ids.shape[0], dtype=cp.uint32), seq_end_col
    ] = 102


def _bert_add_special_tokens_attention_mask(attention_mask, seq_end_col):
    """
    Mark attention mask for special tokens ([CLS] and [SEP]) with 1
    """
    # Copy attention masks for all but last two
    attention_mask[:, 1:-1] = attention_mask[:, 0:-2]
    # Mark [CLS] token with 1
    attention_mask[:, 0] = 1
    # Mark [SEP] token with 1
    attention_mask[
        cp.arange(0, attention_mask.shape[0], dtype=cp.uint32), seq_end_col
    ] = 1


def _bert_add_special_tokens_metadata(metadata, max_length):
    """
    Edit metadata to account for the added special tokens ([CLS] and [SEP])
    """
    # metadata seq starts from plus 1
    metadata[:, 1] = metadata[:, 1] + 1
    # clip done to take overflow into account
    metadata[:, 2] = cp.clip(
        metadata[:, 2] + 1, a_min=None, a_max=max_length - 2
    )
