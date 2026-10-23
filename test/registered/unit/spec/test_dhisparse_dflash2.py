import pytest
import torch

from sglang.srt.models.dflash2 import CandidateSelector, _grouped_conv


def test_selector_matches_explicit_learned_transition_scores_and_path_probabilities():
    torch.manual_seed(12)
    selector = CandidateSelector(hidden_size=3, vocab_size=9, state_rank=2, top_k=3)
    with torch.no_grad():
        selector.predecessor_codebook.normal_()
        selector.successor_codebook.normal_()
    ids = torch.tensor([[[1, 3, 5], [2, 4, 7], [0, 6, 8]]])
    unary, hidden = torch.randn(1, 3, 3), torch.randn(1, 3, 3)
    anchor = torch.tensor([8])
    scores = selector.build_lattice(
        candidate_ids=ids,
        unary_logits=unary,
        hidden_states=hidden,
        anchor_token_ids=anchor,
    )
    projection = selector.hidden_projection(hidden)
    for edge in range(3):
        for predecessor in range(3):
            token = int(anchor[0]) if edge == 0 else int(ids[0, edge - 1, predecessor])
            for candidate in range(3):
                expected = (
                    unary[0, edge, candidate]
                    + (
                        selector.predecessor_codebook[token]
                        * projection[0, edge]
                        * selector.successor_codebook[ids[0, edge, candidate]]
                    ).sum()
                )
                torch.testing.assert_close(
                    scores[0, edge, predecessor, candidate], expected
                )
    uniforms = torch.tensor([[0.15, 0.66, 0.97]])
    for greedy in (False, True):
        tokens, q = selector.sample_path(
            candidate_ids=ids,
            scores=scores,
            uniforms=uniforms,
            temperatures=torch.tensor([0.7]),
            greedy_mask=torch.tensor([greedy]),
        )
        previous = 0
        for edge in range(3):
            row = scores[0, edge, previous]
            probs = torch.softmax(row.float() / 0.7, dim=-1)
            index = (
                row.argmax()
                if greedy
                else (uniforms[0, edge] >= probs.cumsum(-1)).sum().clamp_max(2)
            )
            assert tokens[0, edge] == ids[0, edge, index]
            expected = (
                torch.nn.functional.one_hot(index, 3).float() if greedy else probs
            )
            torch.testing.assert_close(q[0, edge], expected)
            previous = int(index)


@pytest.mark.parametrize("block", [4, 5])
def test_grouped_convolution_never_reads_previous_request(block):
    hidden = torch.arange(2 * block * 6).view(2 * block, 6).float()
    delta = torch.zeros(2 * block, 3, 2)
    base = torch.ones(3, 6)
    result = _grouped_conv(hidden, delta, base, block, 2, 3, 3)
    for row in range(2 * block):
        expected = sum(hidden[row - tap] for tap in range(min(3, row % block + 1)))
        torch.testing.assert_close(result[row], expected)


def test_dflash2_rejects_missing_trained_selector_weights():
    from sglang.srt.models.dflash2 import DFlash2DraftModel

    model = DFlash2DraftModel.__new__(DFlash2DraftModel)
    torch.nn.Module.__init__(model)
    model.candidate_selector = CandidateSelector(
        hidden_size=3, vocab_size=9, state_rank=2, top_k=3
    )
    with pytest.raises(ValueError, match="missing trained"):
        model.load_weights([])
    weights = [
        (name, torch.ones_like(value)) for name, value in model.named_parameters()
    ]
    model.load_weights(weights)
    for _, value in model.named_parameters():
        assert (value == 1).all()
