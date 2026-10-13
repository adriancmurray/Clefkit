"""JointSchemaHead in mlx.nn, mirroring joint_schema_model.JointSchemaHead.

Parameter names match the PyTorch state dict exactly, so joint_head.safetensors
loads with ``load_weights(strict=True)`` and no renaming. Dropout is omitted
(the release runs in eval mode, where it is the identity).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx
import mlx.nn as nn

LOG_100 = math.log(100.0)


class TorchMultiheadAttention(nn.Module):
    """torch.nn.MultiheadAttention (batch_first, no masks, packed in_proj)."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.in_proj_weight = mx.zeros((3 * width, width))
        self.in_proj_bias = mx.zeros((3 * width,))
        self.out_proj = nn.Linear(width, width)

    def __call__(self, query: mx.array, key: mx.array, value: mx.array) -> mx.array:
        width = query.shape[-1]
        w_q, w_k, w_v = mx.split(self.in_proj_weight, 3, axis=0)
        b_q, b_k, b_v = mx.split(self.in_proj_bias, 3, axis=0)
        q = query @ w_q.T + b_q
        k = key @ w_k.T + b_k
        v = value @ w_v.T + b_v
        batch, q_len, _ = q.shape
        kv_len = k.shape[1]
        head_dim = width // self.heads
        q = q.reshape(batch, q_len, self.heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, kv_len, self.heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, kv_len, self.heads, head_dim).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
        attended = mx.softmax(scores, axis=-1, precise=True) @ v
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, q_len, width)
        return self.out_proj(attended)


class EvidenceRoutingLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = TorchMultiheadAttention(width, heads)
        self.feedforward_norm = nn.LayerNorm(width)
        # Same indices as the torch Sequential (Linear, GELU, Dropout, Linear[, Dropout]).
        self.feedforward = [
            nn.Linear(width, feedforward), nn.GELU(), nn.Identity(), nn.Linear(feedforward, width)
        ]

    def _ff(self, x: mx.array) -> mx.array:
        return self.feedforward[3](nn.gelu(self.feedforward[0](x)))

    def __call__(self, queries: mx.array, memory: mx.array) -> mx.array:
        normalized_memory = self.memory_norm(memory)
        routed = self.attention(self.query_norm(queries), normalized_memory, normalized_memory)
        queries = queries + routed
        return queries + self._ff(self.feedforward_norm(queries))


class TransformerDecoderLayer(nn.Module):
    """torch.nn.TransformerDecoderLayer(norm_first=True, activation='gelu'), no masks."""

    def __init__(self, width: int, heads: int, feedforward: int) -> None:
        super().__init__()
        self.self_attn = TorchMultiheadAttention(width, heads)
        self.multihead_attn = TorchMultiheadAttention(width, heads)
        self.linear1 = nn.Linear(width, feedforward)
        self.linear2 = nn.Linear(feedforward, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)

    def __call__(self, x: mx.array, memory: mx.array) -> mx.array:
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h)
        x = x + self.multihead_attn(self.norm2(x), memory, memory)
        return x + self.linear2(nn.gelu(self.linear1(self.norm3(x))))


def _normalize(x: mx.array, eps: float = 1e-12) -> mx.array:
    """torch.nn.functional.normalize(x, dim=-1)."""
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), eps)


def _cosine(a: mx.array, b: mx.array, eps: float = 1e-8) -> mx.array:
    """torch.nn.functional.cosine_similarity(a, b, dim=-1)."""
    a_norm = mx.maximum(mx.linalg.norm(a, axis=-1), eps)
    b_norm = mx.maximum(mx.linalg.norm(b, axis=-1), eps)
    return mx.sum(a * b, axis=-1) / (a_norm * b_norm)


class JointSchemaHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        width: int,
        routing_layers: int,
        layers: int,
        heads: int,
        feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(3, width)
        self.evidence_layers = [
            EvidenceRoutingLayer(width, heads, feedforward) for _ in range(routing_layers)
        ]
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = [TransformerDecoderLayer(width, heads, feedforward) for _ in range(layers)]
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.residual_scorer = [
            nn.Linear(width * 4, width), nn.GELU(), nn.Identity(), nn.Linear(width, 1)
        ]
        self.prior_logit_scale = mx.zeros(())
        self.joint_logit_scale = mx.zeros(())
        self.residual_gate = mx.zeros(())

    @staticmethod
    def _mean_span(values: mx.array, span: Sequence[int]) -> mx.array:
        start, end = span
        return mx.mean(values[start:end], axis=0)

    def __call__(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        attention_mask: mx.array,
        records: Sequence[Any],
        output_embedding_weight: mx.array,
    ) -> list[list[mx.array]]:
        """Same contract as the PyTorch forward: one logit vector per question per record.

        ``hidden_states`` is cast to the head's parameter dtype. ``records`` are
        EncodedRecord-like objects (``questions`` with ``question_type``,
        ``question_span``, ``option_spans``).
        """
        dtype = self.hidden_norm.weight.dtype
        normalized_hidden = self.hidden_norm(hidden_states.astype(dtype))
        results: list[list[mx.array]] = []
        for batch_index, record in enumerate(records):
            sequence_length = int(mx.sum(attention_mask[batch_index]).item())
            sequence_hidden = normalized_hidden[batch_index, :sequence_length]
            memory = self.memory_projection(sequence_hidden)[None]
            global_vector = sequence_hidden[-1]
            question_vectors = mx.stack(
                [self._mean_span(sequence_hidden, q.question_span) for q in record.questions]
            )
            type_ids = mx.array([q.question_type for q in record.questions])

            option_contexts: list[mx.array] = []
            lexical_options: list[mx.array] = []
            option_counts: list[int] = []
            for question in record.questions:
                option_contexts.append(
                    mx.stack([self._mean_span(sequence_hidden, s) for s in question.option_spans])
                )
                lexical = []
                for start, end in question.option_spans:
                    token_ids = input_ids[batch_index, start:end]
                    rows = output_embedding_weight[token_ids].astype(dtype)
                    lexical.append(mx.mean(rows, axis=0))
                lexical_options.append(mx.stack(lexical))
                option_counts.append(len(question.option_spans))

            option_queries = [
                self.option_context_projection(context)
                + self.option_lexical_projection(lexical)
                + self.option_question_projection(question_vectors[index])[None]
                for index, (context, lexical) in enumerate(zip(option_contexts, lexical_options))
            ]
            routed_options = mx.concatenate(option_queries, axis=0)[None]
            for layer in self.evidence_layers:
                routed_options = layer(routed_options, memory)
            routed_options = routed_options[0]
            boundaries = [sum(option_counts[: i + 1]) for i in range(len(option_counts) - 1)]
            split_options = mx.split(routed_options, boundaries, axis=0) if boundaries else [routed_options]

            base_fields = self.question_projection(question_vectors)
            option_summaries = []
            for index, options in enumerate(split_options):
                field = base_fields[index]
                weights = mx.softmax(
                    (options @ field) / math.sqrt(options.shape[-1]), axis=0, precise=True
                )
                option_summaries.append(mx.sum(weights[:, None] * options, axis=0))
            fields = (
                base_fields
                + self.option_summary_norm(mx.stack(option_summaries))
                + self.global_projection(global_vector)[None]
                + self.type_embedding(type_ids)
            )[None]
            for layer in self.layers:
                fields = layer(fields, memory)
            fields = self.field_norm(fields[0])

            prior_scale = mx.exp(mx.minimum(self.prior_logit_scale, LOG_100))
            joint_scale = mx.exp(mx.minimum(self.joint_logit_scale, LOG_100))
            gate = mx.sigmoid(self.residual_gate)
            record_logits: list[mx.array] = []
            for index, (lexical, routed) in enumerate(zip(lexical_options, split_options)):
                field = fields[index]
                anchor = _normalize(question_vectors[index] + global_vector)
                prior = prior_scale * (_normalize(lexical) @ anchor)
                options = self.option_norm(routed)
                repeated = mx.broadcast_to(field[None], options.shape)
                cosine = _cosine(repeated, options)
                features = mx.concatenate(
                    [repeated, options, repeated * options, mx.abs(repeated - options)], axis=-1
                )
                residual = self.residual_scorer[3](
                    nn.gelu(self.residual_scorer[0](features))
                ).squeeze(-1)
                record_logits.append(prior + gate * (joint_scale * cosine + residual))
            results.append(record_logits)
        return results


def load_head(model_dir: str | Path, dtype: mx.Dtype = mx.float32) -> JointSchemaHead:
    """Load joint_head_config.json + joint_head.safetensors (strict, never quantized).

    The release stores the head in BF16; ``dtype`` controls compute precision
    (float32 default; pass mx.bfloat16 to mirror load_release_model exactly).
    """
    path = Path(model_dir)
    config = json.loads((path / "joint_head_config.json").read_text())
    head = JointSchemaHead(**config)
    weights = mx.load(str(path / "joint_head.safetensors"))
    head.load_weights([(k, v.astype(dtype)) for k, v in weights.items()], strict=True)
    head.set_dtype(dtype)
    head.eval()
    mx.eval(head.parameters())
    return head
