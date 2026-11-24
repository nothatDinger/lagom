import json
import random
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import (
    ASSISTANT_SUFFIX,
    SHAREGPT_FILENAME,
    SHAREGPT_REPO_ID,
    BaseDataset,
    DatasetRow,
)
from sglang.benchmark.sharegpt_packing import pack_sharegpt
from sglang.benchmark.utils import (
    download_and_cache_hf_file,
    is_file_valid_json,
    remove_suffix,
)


@dataclass
class ShareGPTDataset(BaseDataset):
    dataset_path: str
    num_requests: int
    fixed_output_len: Optional[int]
    context_len: Optional[int]
    prompt_suffix: str
    apply_chat_template: bool
    fixed_input_len: Optional[int] = None
    seed: int = 0

    @classmethod
    def from_args(cls, args: Namespace) -> "ShareGPTDataset":
        length = getattr(args, "sharegpt_input_len", None)
        if length is None:
            assert not getattr(args, "tokenize_prompt", False)
        elif not (
            args.dataset_path
            and args.tokenize_prompt
            and length > 0
            and args.sharegpt_output_len
        ):
            raise ValueError(
                "Fixed ShareGPT inputs require a local dataset, positive lengths and --tokenize-prompt"
            )
        elif args.apply_chat_template or args.prompt_suffix:
            raise ValueError(
                "Fixed ShareGPT inputs pack raw conversations without a chat template or prompt suffix"
            )
        return cls(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            fixed_output_len=args.sharegpt_output_len,
            context_len=args.sharegpt_context_len,
            prompt_suffix=args.prompt_suffix,
            apply_chat_template=args.apply_chat_template,
            fixed_input_len=length,
            seed=getattr(args, "seed", 0),
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        if self.fixed_input_len is not None:
            if (
                self.fixed_output_len is None
                or self.fixed_output_len < 1
                or (
                    self.context_len
                    and self.fixed_input_len + self.fixed_output_len > self.context_len
                )
            ):
                raise ValueError(
                    "Invalid fixed ShareGPT input/output length for the requested context"
                )
            with open(self.dataset_path) as f:
                packed = pack_sharegpt(
                    json.load(f),
                    lambda text: tokenizer.encode(text, add_special_tokens=False),
                    count=self.num_requests,
                    length=self.fixed_input_len,
                    seed=self.seed,
                )
            return [
                DatasetRow(
                    prompt=row["input_ids"],
                    prompt_len=self.fixed_input_len,
                    output_len=self.fixed_output_len,
                )
                for row in packed
            ]
        return sample_sharegpt_requests(
            dataset_path=self.dataset_path,
            num_requests=self.num_requests,
            tokenizer=tokenizer,
            fixed_output_len=self.fixed_output_len,
            context_len=self.context_len,
            prompt_suffix=self.prompt_suffix,
            apply_chat_template=self.apply_chat_template,
        )


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    context_len: Optional[int] = None,
    prompt_suffix: Optional[str] = "",
    apply_chat_template=False,
) -> List[DatasetRow]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Download sharegpt if necessary
    if not is_file_valid_json(dataset_path) and dataset_path == "":
        dataset_path = download_and_cache_hf_file(
            repo_id=SHAREGPT_REPO_ID,
            filename=SHAREGPT_FILENAME,
        )

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Filter out the conversations with less than 2 turns.
    dataset = [
        data
        for data in dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]
    # Only keep the first two turns of each conversation.
    dataset = [
        (
            data.get("conversations", data.get("conversation", []))[0]["value"],
            data.get("conversations", data.get("conversation", []))[1]["value"],
        )
        for data in dataset
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[DatasetRow] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        if prompt_suffix:
            prompt = (
                remove_suffix(prompt, ASSISTANT_SUFFIX)
                + prompt_suffix
                + ASSISTANT_SUFFIX
            )

        if apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                return_dict=False,
            )
            if tokenizer.bos_token:
                prompt = prompt.replace(tokenizer.bos_token, "")

        prompt_token_ids = tokenizer.encode(prompt)
        completion = dataset[i][1]
        completion_token_ids = tokenizer.encode(completion)
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )

        if prompt_len < 2 or output_len < 2:
            # Prune too short sequences.
            continue

        if context_len and prompt_len + output_len > context_len:
            # Prune too long sequences.
            continue

        filtered_dataset.append(
            DatasetRow(
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
            )
        )

    print(f"#Input tokens: {np.sum([x.prompt_len for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in filtered_dataset])}")
    return filtered_dataset
