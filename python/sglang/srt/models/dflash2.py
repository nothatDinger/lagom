"""DFlash2 checkpoint support, adapted from SGLang commit 6e667c431009.

Preserves the learned transition scores, proposal probabilities and grouped
convolutions. The worker verifies the sampled proposal using rejection sampling.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed import get_tp_group, tensor_model_parallel_all_gather
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.speculative.dflash_utils import _get_dflash_config


def _grouped_conv(hidden_states, delta, base, block_size, num_groups, group_size, taps):
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    out = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        shifted = torch.where((position >= tap).view(-1, 1, 1), shifted, 0)
        out = out + coefficients[:, tap] * shifted
    return out.flatten(-2)


class DFlashGroupedConv(nn.Module):
    """Grouped dynamic depthwise K-tap convolution across one DFlash block.

    Each sublayer is wrapped: `prepare` convolves its input and returns the kernel
    for `finish` to convolve its output, both from one projection of the input.
    """

    def __init__(
        self, hidden_size: int, block_size: int, taps: int, group_size: int
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"DFLASH conv_group_size={group_size} must divide "
                f"hidden_size={hidden_size}."
            )
        hidden_size = int(hidden_size)
        self.block_size = int(block_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = hidden_size // self.group_size
        # [input/output, tap, channel], the layout training exports.
        base_kernel = torch.zeros(2, self.taps, hidden_size)
        base_kernel[:, 0] = 1.0
        self.base_kernel = nn.Parameter(base_kernel)
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * self.taps * self.num_groups, bias=False
        )

    def _convolve(self, hidden_states, delta, side: int) -> torch.Tensor:
        # Marked here, not inside: by the time the compiled function traces, the dim
        # is symbolic and the group index costs an integer div and mod per element.
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor):
        coefficients = self.kernel_projection(hidden_states).reshape(
            *hidden_states.shape[:-1], 2, self.taps, self.num_groups
        )
        return (
            self._convolve(hidden_states, coefficients[..., 0, :, :], side=0),
            coefficients[..., 1, :, :],
        )

    def finish(self, hidden_states: torch.Tensor, coefficients) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, side=1)


def _score_edges(
    *,
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    keys = successor_table[candidate_ids]
    # Concatenate the ids and look them up once. Concatenating the looked-up rows
    # instead moves a [b, slots, k, rank] float tensor where this moves one id per
    # candidate, and it costs a second gather for the anchor.
    predecessor_ids = torch.cat(
        [anchor_token_ids[:, None, None].expand(-1, 1, top_k), candidate_ids[:, :-1]],
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], keys
    )


def _follow_maps(maps, initial_indices, edges: int):
    index = initial_indices
    path = [index]
    for edge in range(edges):
        index = maps[:, edge].gather(-1, index[:, None])[:, 0]
        path.append(index)
    return torch.stack(path, dim=1)


class CandidateSelector(nn.Module):
    """Scores the K x K transitions between adjacent proposal slots, then walks them.

    The [vocab, r] tables are replicated on every TP rank rather than sharded like
    the LM head: candidate ids are gathered globally, so any rank can need any row.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        state_rank: int,
        top_k: int,
    ) -> None:
        super().__init__()
        state_rank = int(state_rank)
        self.top_k = int(top_k)
        self.predecessor_codebook = nn.Parameter(
            torch.zeros(int(vocab_size), state_rank), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.zeros(int(vocab_size), state_rank), requires_grad=False
        )
        self.hidden_projection = nn.Linear(hidden_size, state_rank, bias=False)

    def build_lattice(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """score[b,e,p,c] = unary[b,e,c] + <A[pred[b,e,p]] * project(h[b,e]), B[c]>

        pred is cand[b,e-1], and the verified anchor for slot 0.
        """
        # Everything but the batch is a model constant. Left symbolic, inductor
        # recovers indices with an integer division per element instead of folding.
        hidden = self.hidden_projection(hidden_states)
        return _score_edges(
            predecessor_table=self.predecessor_codebook,
            successor_table=self.successor_codebook,
            candidate_ids=candidate_ids,
            unary_logits=unary_logits,
            hidden=hidden,
            anchor_token_ids=anchor_token_ids,
            top_k=self.top_k,
        )

    def sample_path(
        self,
        *,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        uniforms: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Walk one path, with q over the K candidates for the verify. greedy_mask
        rows take the argmax, selected rather than branched, so one captured graph
        serves greedy and sampling batches alike."""
        top_k = self.top_k
        temps = temperatures.view(-1, 1)
        initial_probs = torch.softmax(scores[:, 0, 0].float() / temps, dim=-1)
        initial_indices = (
            uniforms[:, :1]
            .ge(initial_probs.cumsum(dim=-1))
            .sum(dim=-1)
            .clamp_max(top_k - 1)
        )
        transition_probs = torch.softmax(
            scores[:, 1:].float() / temps[:, :, None, None], dim=-1
        )
        local_maps = (
            uniforms[:, 1:, None, None]
            .ge(transition_probs.cumsum(dim=-1))
            .sum(dim=-1)
            .clamp_max(top_k - 1)
        )
        initial_indices = torch.where(
            greedy_mask, scores[:, 0, 0].argmax(dim=-1), initial_indices
        )
        local_maps = torch.where(
            greedy_mask[:, None, None], scores[:, 1:].argmax(dim=-1), local_maps
        )
        path_indices = _follow_maps(
            local_maps, initial_indices, int(scores.shape[1]) - 1
        )
        tokens = candidate_ids.gather(-1, path_indices.unsqueeze(-1))[:, :, 0]
        realized_rows = transition_probs.gather(
            2, path_indices[:, :-1, None, None].expand(-1, -1, 1, top_k)
        )[:, :, 0]
        q_rows = torch.cat((initial_probs.unsqueeze(1), realized_rows), dim=1)
        # Greedy rows walk the argmax, so their q is the point mass there, not
        # the temperature-1 softmax above. The triton walk stores the same.
        q_rows = torch.where(
            greedy_mask[:, None, None], F.one_hot(path_indices, top_k).float(), q_rows
        )
        return tokens, q_rows


class DFlash2DraftModel(DFlashDraftModel):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        cfg = _get_dflash_config(config)
        rank, top_k = cfg.get("selector_rank", 0), cfg.get("selector_top_k", 0)
        if any(type(v) is not int or v <= 0 for v in (rank, top_k)):
            raise ValueError(
                "DFlash2 requires positive selector_rank and selector_top_k"
            )
        self.candidate_selector = CandidateSelector(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            state_rank=rank,
            top_k=top_k,
        )
        self.output_multiplier = float(cfg.get("output_multiplier", 1.0))
        self.softcap = float(cfg.get("final_logit_softcapping") or 0)
        if self.output_multiplier <= 0 or self.softcap < 0:
            raise ValueError("Invalid DFlash2 unary-logit transform")
        taps, group = cfg.get("conv_kernel_size", 0), cfg.get("conv_group_size", 0)
        if bool(taps) != bool(group) or min(taps, group) < 0:
            raise ValueError(
                "DFlash2 convolution requires kernel size and group size together"
            )
        if taps:
            for layer in self.layers:
                layer.attention_conv = DFlashGroupedConv(
                    config.hidden_size, self.block_size, taps, group
                )
                layer.mlp_conv = DFlashGroupedConv(
                    config.hidden_size, self.block_size, taps, group
                )
        # The shared head is attached after loading the draft checkpoint.
        self.lm_head = None

    def set_block_size(self, block_size):
        self.block_size = int(block_size)
        for layer in self.layers:
            for conv in (layer.attention_conv, layer.mlp_conv):
                if conv is not None:
                    conv.block_size = self.block_size

    def load_weights(self, weights):
        required = {
            name
            for name, _ in self.named_parameters()
            if name.startswith("candidate_selector.")
            or ".attention_conv." in name
            or ".mlp_conv." in name
        }
        loaded = set()

        def checked_weights():
            for name, weight in weights:
                loaded.add(name.removeprefix("model."))
                yield name, weight

        super().load_weights(checked_weights())
        if missing := required - loaded:
            raise ValueError(
                f"DFlash2 checkpoint is missing trained selector/convolution weights: {sorted(missing)}"
            )

    def compute_candidates(self, hidden):
        head = self.lm_head
        weight = getattr(head, "weight", None)
        if weight is None or weight.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError(
                "DFlash2 candidate projection requires a dense floating-point target head"
            )
        shard = getattr(head, "shard_indices", None)
        count = (
            int(shard.num_org_elements)
            if shard is not None
            else int(self.config.vocab_size)
        )
        offset = int(shard.org_vocab_start_index) if shard is not None else 0
        k = self.candidate_selector.top_k
        logits = F.linear(hidden.to(weight.dtype), weight[:count]).float()
        vals, ids = torch.topk(logits, min(k, count), dim=-1)
        ids = ids + offset
        group = get_tp_group()
        if group.world_size > 1:
            vals = tensor_model_parallel_all_gather(vals, dim=-1)
            ids = tensor_model_parallel_all_gather(ids, dim=-1)
        vals, order = torch.topk(vals, k, dim=-1)
        ids = ids.gather(-1, order)
        vals = vals * self.output_multiplier
        if self.softcap:
            vals = torch.tanh(vals / self.softcap) * self.softcap
        return ids, vals


EntryClass = [DFlash2DraftModel]
