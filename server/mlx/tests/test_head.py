from types import SimpleNamespace

import mlx.core as mx
import pytest
from conftest import MODEL_DIR, TINY_HEAD, needs_release

from clefkit_mlx.head import JointSchemaHead, load_head


def _record(offset=0):
    q = lambda t, qs, opts: SimpleNamespace(question_type=t, question_span=qs, option_spans=opts)  # noqa: E731
    return SimpleNamespace(
        questions=(
            q(1, (offset + 2, offset + 5), ((offset + 6, offset + 8), (offset + 8, offset + 9), (offset + 10, offset + 12))),
            q(0, (offset + 12, offset + 14), ((offset + 14, offset + 16), (offset + 16, offset + 18))),
        )
    )


def _head():
    mx.random.seed(1)
    head = JointSchemaHead(**TINY_HEAD)
    from mlx.utils import tree_flatten

    head.load_weights([(k, 0.2 * mx.random.normal(v.shape)) for k, v in tree_flatten(head.parameters())])
    head.eval()
    return head


def test_forward_shapes():
    head = _head()
    length, vocab = 24, 50
    hidden = mx.random.normal((1, length, TINY_HEAD["hidden_size"]))
    ids = mx.random.randint(0, vocab, (1, length))
    table = mx.random.normal((vocab, TINY_HEAD["hidden_size"]))
    out = head(hidden, ids, mx.ones((1, length), dtype=mx.int32), [_record()], table)
    assert len(out) == 1 and [x.shape for x in out[0]] == [(3,), (2,)]
    assert all(bool(mx.all(mx.isfinite(x))) for x in out[0])


def test_padding_is_masked():
    """A right-padded record in a batch scores exactly as it does alone, whatever the pad states hold."""
    head = _head()
    short, long, vocab, width = 20, 30, 50, TINY_HEAD["hidden_size"]
    hidden_short = mx.random.normal((1, short, width))
    ids_short = mx.random.randint(0, vocab, (1, short))
    table = mx.random.normal((vocab, width))
    alone = head(hidden_short, ids_short, mx.ones((1, short), dtype=mx.int32), [_record()], table)[0]

    garbage = 100.0 * mx.random.normal((1, long - short, width))
    hidden = mx.concatenate(
        [mx.concatenate([hidden_short, garbage], axis=1), mx.random.normal((1, long, width))], axis=0
    )
    ids = mx.concatenate(
        [mx.concatenate([ids_short, mx.zeros((1, long - short), dtype=ids_short.dtype)], axis=1),
         mx.random.randint(0, vocab, (1, long))], axis=0
    )
    mask = mx.array([[1] * short + [0] * (long - short), [1] * long], dtype=mx.int32)
    batched = head(hidden, ids, mask, [_record(), _record(offset=4)], table)
    for a, b in zip(alone, batched[0]):
        assert mx.allclose(a, b, atol=1e-5).item()
    assert [x.shape for x in batched[1]] == [(3,), (2,)]


def test_prior_only_when_gate_closed():
    """With the residual gate at -inf the logits reduce to the lexical prior (cosine to the anchor)."""
    head = _head()
    head.residual_gate = mx.array(-1e9)
    head.prior_logit_scale = mx.array(0.0)
    length, vocab, width = 24, 50, TINY_HEAD["hidden_size"]
    hidden = mx.random.normal((1, length, width))
    ids = mx.random.randint(0, vocab, (1, length))
    table = mx.random.normal((vocab, width))
    out = head(hidden, ids, mx.ones((1, length), dtype=mx.int32), [_record()], table)[0]
    normed = head.hidden_norm(hidden[0])
    q = _record().questions[0]
    anchor = mx.mean(normed[q.question_span[0]:q.question_span[1]], axis=0) + normed[-1]
    anchor = anchor / mx.linalg.norm(anchor)
    for (start, end), logit in zip(q.option_spans, out[0].tolist()):
        lex = mx.mean(table[ids[0, start:end]], axis=0)
        expected = (lex / mx.linalg.norm(lex)) @ anchor
        assert abs(expected.item() - logit) < 1e-4


@needs_release
def test_real_head_loads_strict():
    head = load_head(MODEL_DIR)
    assert head.hidden_norm.weight.shape == (4096,)
    assert head.hidden_norm.weight.dtype == mx.float32
    bf16 = load_head(MODEL_DIR, dtype=mx.bfloat16)
    assert bf16.memory_projection.weight.dtype == mx.bfloat16
