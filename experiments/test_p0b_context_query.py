"""No-GPU contract tests for the preregistered E3-v2 P0-B execution path."""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from experiments.phase2.e3_v2.context_query import (
    CacheContractError,
    ConsumedStateError,
    InterventionResult,
    RecurrentStateMutationError,
    TokenBoundaryError,
    full_kv_intervention,
    generate_greedy,
    reference_full_prompt_logits,
    run_post_intervention_prompt,
    score_gold_answer_logprob,
    tokenize_answer_continuation,
    tokenize_prompt_parts,
)
from experiments.phase2.e3_v2.protocol import (
    P0B_EXECUTION_EVENTS,
    P0B_PROTOCOL_VERSION,
)
from experiments.utils.cache_access import get_cache_layer
from experiments.utils.dataset_utils import EvalSample
from experiments.utils.eval_harness import PromptTextParts, build_prompt, build_prompt_parts
from experiments.utils.run_manifest import metric_protocol


class CharTokenizer:
    eos_token_id = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        del tokenize, kwargs
        content = messages[0]["content"]
        suffix = "</user><assistant>" if add_generation_prompt else "</user>"
        return f"<user>{content}{suffix}"

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_offsets_mapping=False,
        return_tensors=None,
    ):
        del add_special_tokens
        ids = [ord(char) % 29 + 1 for char in text]
        result = {"input_ids": torch.tensor([ids], dtype=torch.long)}
        if return_offsets_mapping:
            result["offset_mapping"] = torch.tensor(
                [[[index, index + 1] for index in range(len(text))]],
                dtype=torch.long,
            )
        if return_tensors not in (None, "pt"):
            raise ValueError("The test tokenizer supports PyTorch tensors only")
        return result

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(int(token)) for token in token_ids)


class CrossingTokenizer(CharTokenizer):
    def __call__(self, text, **kwargs):
        del text, kwargs
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "offset_mapping": torch.tensor([[[0, 1], [1, 3], [3, 4]]]),
        }


class FakeAttentionLayer:
    def __init__(self):
        self.keys = torch.empty((1, 1, 0, 1), dtype=torch.float32)
        self.values = torch.empty((1, 1, 0, 1), dtype=torch.float32)


class FakeRecurrentLayer:
    def __init__(self):
        self.conv_states = torch.zeros(1)
        self.recurrent_states = torch.zeros(1)
        self.has_previous_state = False
        self.is_conv_states_initialized = False
        self.is_recurrent_states_initialized = False


class FakeHybridCache:
    def __init__(self):
        self.layers = [FakeAttentionLayer(), FakeRecurrentLayer()]


class FakeHybridBaseModel:
    def __init__(self, trace, vocab_size=32):
        self.trace = trace
        self.vocab_size = vocab_size

    def __call__(
        self,
        input_ids,
        *,
        past_key_values=None,
        use_cache=True,
        position_ids=None,
        cache_position=None,
        return_dict=True,
    ):
        del use_cache, return_dict
        is_prefill = past_key_values is None
        cache = FakeHybridCache() if is_prefill else past_key_values
        positions = None if position_ids is None else position_ids.detach().cpu().tolist()[0]
        resident = None if cache_position is None else cache_position.detach().cpu().tolist()
        self.trace.append(
            {
                "kind": "prefill" if is_prefill else "continuation",
                "tokens": input_ids.detach().cpu().tolist()[0],
                "positions": positions,
                "cache_position": resident,
            }
        )

        attention = cache.layers[0]
        recurrent = cache.layers[1]
        hidden_steps = []
        for token in input_ids[0]:
            value = token.to(dtype=torch.float32).reshape(1, 1, 1, 1)
            attention.keys = torch.cat([attention.keys, value], dim=-2)
            attention.values = torch.cat([attention.values, value * 2], dim=-2)
            recurrent.conv_states += value.reshape(1)
            recurrent.recurrent_states += value.reshape(1) * 3
            recurrent.has_previous_state = True
            recurrent.is_conv_states_initialized = True
            recurrent.is_recurrent_states_initialized = True

            attention_sum = attention.keys.sum() + attention.values.sum()
            recurrent_sum = recurrent.conv_states.sum() + recurrent.recurrent_states.sum()
            winner = int((attention_sum + recurrent_sum).item()) % self.vocab_size
            logits = torch.arange(self.vocab_size, dtype=torch.float32) / 100.0
            logits[winner] += 10.0
            hidden_steps.append(logits)

        return SimpleNamespace(
            last_hidden_state=torch.stack(hidden_steps, dim=0).unsqueeze(0),
            past_key_values=cache,
        )


class FakeHybridModel:
    def __init__(self, trace=None):
        self.device = torch.device("cpu")
        self.trace = [] if trace is None else trace
        self.model = FakeHybridBaseModel(self.trace)
        self.lm_head = torch.nn.Identity()


def make_split(tokenizer=None):
    tokenizer = tokenizer or CharTokenizer()
    return tokenize_prompt_parts(tokenizer, PromptTextParts("abc", "?q"))


def keep_context_positions(*positions, trace=None):
    def intervene(cache, context_ids):
        del context_ids
        if trace is not None:
            trace.append({"kind": "intervention"})
        indices = torch.tensor(positions, dtype=torch.long)
        layer = get_cache_layer(cache, 0)
        layer.keys = layer.keys.index_select(-2, indices)
        layer.values = layer.values.index_select(-2, indices)
        return InterventionResult("keep", indices)

    return intervene


class PromptBoundaryTests(unittest.TestCase):
    def test_chat_prompt_parts_reconstruct_existing_prompt(self):
        tokenizer = CharTokenizer()
        sample = EvalSample(
            dataset="needle",
            sample_id="case",
            context="memory",
            question="where",
            answer="here",
            context_length=64,
        )
        parts = build_prompt_parts(sample, tokenizer)
        self.assertEqual(parts.full_prompt, build_prompt(sample, tokenizer))
        self.assertTrue(parts.memory_context.startswith("<user>memory"))
        self.assertTrue(parts.query_suffix.endswith("</user><assistant>"))

    def test_single_tokenization_reconstructs_exact_prompt(self):
        split = make_split()
        self.assertTrue(torch.equal(torch.cat([split.context_ids, split.query_ids], 1), split.full_ids))
        self.assertEqual(split.split_token_index, split.context_tokens)

    def test_boundary_crossing_token_fails_closed(self):
        with self.assertRaises(TokenBoundaryError):
            tokenize_prompt_parts(CrossingTokenizer(), PromptTextParts("ab", "cd"))

    def test_gold_continuation_preserves_prompt_prefix(self):
        tokenizer = CharTokenizer()
        split = make_split(tokenizer)
        answer_ids = tokenize_answer_continuation(tokenizer, split, "answer")
        self.assertGreater(answer_ids.shape[1], 1)


class ExecutionContractTests(unittest.TestCase):
    def run_arm(self, intervention=full_kv_intervention, trace=None):
        model = FakeHybridModel(trace)
        state = run_post_intervention_prompt(
            model,
            make_split(),
            attention_layer_indices=[0],
            recurrent_layer_indices=[1],
            intervention=intervention,
        )
        return model, state

    def test_context_intervention_query_order_and_logical_positions(self):
        trace = []
        model, state = self.run_arm(keep_context_positions(0, 2, trace=trace), trace)
        self.assertEqual([entry["kind"] for entry in trace], ["prefill", "intervention", "continuation"])
        self.assertEqual(trace[-1]["positions"], list(range(3, 3 + make_split().query_tokens)))
        self.assertEqual(trace[-1]["cache_position"], list(range(2, 2 + make_split().query_tokens)))
        self.assertEqual(state.events, P0B_EXECUTION_EVENTS)
        self.assertEqual(len(model.trace), 3)

    def test_full_kv_split_matches_unmodified_full_prompt(self):
        model = FakeHybridModel()
        split = make_split()
        reference = reference_full_prompt_logits(model, split.full_ids)
        state = run_post_intervention_prompt(
            model,
            split,
            attention_layer_indices=[0],
            recurrent_layer_indices=[1],
        )
        torch.testing.assert_close(state.first_answer_logits, reference)
        self.assertEqual(
            [entry["kind"] for entry in model.trace],
            ["prefill", "prefill", "continuation"],
        )

    def test_attention_intervention_changes_logits_without_recurrent_edit(self):
        _, full_state = self.run_arm()
        _, reduced_state = self.run_arm(keep_context_positions(0, 2))
        self.assertFalse(torch.equal(full_state.first_answer_logits, reduced_state.first_answer_logits))
        self.assertEqual(reduced_state.active_context_positions.tolist(), [0, 2])

    def test_recurrent_mutation_is_rejected_before_query(self):
        trace = []

        def bad_intervention(cache, context_ids):
            cache.layers[1].recurrent_states += 1
            return full_kv_intervention(cache, context_ids)

        with self.assertRaises(RecurrentStateMutationError):
            self.run_arm(bad_intervention, trace)
        self.assertEqual([entry["kind"] for entry in trace], ["prefill"])

    def test_active_position_count_must_match_resident_kv(self):
        def lying_intervention(cache, context_ids):
            del cache, context_ids
            return InterventionResult("lying", torch.tensor([0, 1]))

        with self.assertRaises(CacheContractError):
            self.run_arm(lying_intervention)

    def test_identical_arms_have_identical_logits_and_greedy_output(self):
        tokenizer = CharTokenizer()
        model_a, state_a = self.run_arm(keep_context_positions(0, 2))
        model_b, state_b = self.run_arm(keep_context_positions(0, 2))
        torch.testing.assert_close(state_a.first_answer_logits, state_b.first_answer_logits)
        output_a = generate_greedy(model_a, tokenizer, state_a, max_new_tokens=4)
        output_b = generate_greedy(model_b, tokenizer, state_b, max_new_tokens=4)
        self.assertTrue(torch.equal(output_a.token_ids, output_b.token_ids))
        self.assertEqual(output_a.text, output_b.text)

    def test_gold_score_starts_from_post_intervention_logits(self):
        model, state = self.run_arm(keep_context_positions(0, 2))
        targets = torch.tensor([[int(state.first_answer_logits.argmax()), 4]])
        expected_first = torch.log_softmax(state.first_answer_logits, -1)[0, targets[0, 0]].item()
        result = score_gold_answer_logprob(model, state, targets)
        self.assertEqual(result.n_tokens, 2)
        self.assertAlmostEqual(result.token_logprobs[0], expected_first)
        self.assertAlmostEqual(result.mean_logprob, result.total_logprob / 2)
        self.assertTrue(math.isfinite(result.mean_logprob))

    def test_post_intervention_state_is_single_use(self):
        model, state = self.run_arm()
        score_gold_answer_logprob(model, state, torch.tensor([[1]]))
        with self.assertRaises(ConsumedStateError):
            generate_greedy(model, CharTokenizer(), state, max_new_tokens=1)

    def test_invalid_answer_does_not_consume_state(self):
        model, state = self.run_arm()
        with self.assertRaisesRegex(RuntimeError, "non-empty"):
            score_gold_answer_logprob(model, state, torch.empty((1, 0), dtype=torch.long))
        self.assertIsNone(state.consumed_by)


class ProtocolManifestTests(unittest.TestCase):
    def test_manifest_pins_post_intervention_protocol(self):
        protocol = metric_protocol()["post_intervention"]
        self.assertEqual(protocol["version"], P0B_PROTOCOL_VERSION)
        self.assertEqual(protocol["execution_order"], list(P0B_EXECUTION_EVENTS))
        self.assertEqual(protocol["primary_quality"], "mean_gold_answer_logprob_per_token")


if __name__ == "__main__":
    unittest.main()
