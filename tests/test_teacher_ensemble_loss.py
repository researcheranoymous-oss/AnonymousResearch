import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torch.nn.functional as F


def load_trainer_method(name):
    """Load a trainer method without importing optional training dependencies."""
    source = Path(__file__).parents[1].joinpath("mass_trainer.py").read_text()
    module = ast.parse(source)
    trainer_class = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "MASSTrainer")
    function = next(
        node
        for node in trainer_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    function.decorator_list = []
    namespace = {
        "torch": torch,
        "F": F,
        "nullcontext": nullcontext,
        "is_peft_model": lambda model: False,
        "empty_cache": lambda: None,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "mass_trainer.py", "exec"), namespace)
    return namespace[name]


class TeacherEnsembleLossTest(unittest.TestCase):
    def test_forward_kl_uses_averaged_teacher_probabilities(self):
        loss_fn = load_trainer_method("generalized_jsd_loss")
        temperature = 1.3
        student_logits = torch.tensor([[[0.2, -0.4, 0.7]]], dtype=torch.float64)
        teacher_logits_a = torch.tensor([[[1.2, 0.1, -0.5]]], dtype=torch.float64)
        teacher_logits_b = torch.tensor([[[-0.3, 1.0, 0.4]]], dtype=torch.float64)
        teacher_probs = (
            F.softmax(teacher_logits_a / temperature, dim=-1)
            + F.softmax(teacher_logits_b / temperature, dim=-1)
        ) / 2

        actual = loss_fn(
            student_logits,
            None,
            labels=torch.tensor([[1]]),
            beta=0,
            temperature=temperature,
            teacher_probs=teacher_probs,
        )
        expected = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            torch.log(teacher_probs),
            reduction="none",
            log_target=True,
        ).sum()
        self.assertTrue(torch.allclose(actual, expected))

    def test_top_k_is_selected_from_averaged_distribution(self):
        loss_fn = load_trainer_method("generalized_jsd_loss")
        student_logits = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]], dtype=torch.float64)
        teacher_probs = torch.tensor([[[0.35, 0.40, 0.20, 0.05]]], dtype=torch.float64)

        actual = loss_fn(
            student_logits,
            None,
            labels=torch.tensor([[1]]),
            beta=0,
            teacher_probs=teacher_probs,
            top_k=2,
        )
        indices = torch.tensor([[[1, 0]]])
        selected_student = torch.gather(student_logits, -1, indices)
        selected_teacher = torch.gather(teacher_probs, -1, indices)
        selected_teacher = selected_teacher / selected_teacher.sum(dim=-1, keepdim=True)
        expected = F.kl_div(
            F.log_softmax(selected_student, dim=-1),
            torch.log(selected_teacher),
            reduction="none",
            log_target=True,
        ).sum()
        self.assertTrue(torch.allclose(actual, expected))

    def test_compute_loss_forwards_each_view_and_averages_predictions(self):
        loss_fn = load_trainer_method("generalized_jsd_loss")
        compute_loss = load_trainer_method("compute_loss")
        student_target_logits = torch.tensor([0.1, 0.4, -0.2], dtype=torch.float64)
        teacher_target_logits = [
            torch.tensor([1.0, -0.2, 0.1], dtype=torch.float64),
            torch.tensor([-0.4, 1.2, 0.3], dtype=torch.float64),
            torch.tensor([0.2, 0.1, 1.1], dtype=torch.float64),
        ]

        class FakeModel:
            def __init__(self):
                self.calls = 0

            def __call__(self, input_ids, attention_mask):
                target = student_target_logits if self.calls == 0 else teacher_target_logits[self.calls - 1]
                self.calls += 1
                logits = torch.zeros((1, 3, 3), dtype=torch.float64)
                logits[:, 1, :] = target
                if self.calls == 1:
                    logits.requires_grad_()
                return SimpleNamespace(logits=logits)

        trainer = SimpleNamespace(
            use_thinking_machines_loss=False,
            temperature=1.0,
            use_ema_teacher=False,
            fixed_teacher=False,
            beta=0,
            top_k_loss=None,
            jsd_token_clip=None,
            generalized_jsd_loss=loss_fn,
        )
        trainer.compute_loss = MethodType(compute_loss, trainer)
        inputs = {
            "student_prompt_length": 2,
            "teacher_prompt_length": 2,
            "student_input_ids": torch.tensor([[7, 8, 1]]),
            "student_attention_mask": torch.ones((1, 3), dtype=torch.long),
            "teacher_input_ids": torch.ones((3, 1, 3), dtype=torch.long),
            "teacher_attention_mask": torch.ones((3, 1, 3), dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 1]]),
        }
        model = FakeModel()

        actual = trainer.compute_loss(model, inputs)
        teacher_probs = torch.stack(
            [F.softmax(logits, dim=-1) for logits in teacher_target_logits]
        ).mean(dim=0).reshape(1, 1, 3)
        expected = loss_fn(
            student_target_logits.reshape(1, 1, 3),
            None,
            labels=torch.tensor([[1]]),
            beta=0,
            teacher_probs=teacher_probs,
        )
        self.assertEqual(model.calls, 4)
        self.assertTrue(torch.allclose(actual, expected))

    def test_selective_weight_matches_eq5_closed_form(self):
        weight_fn = load_trainer_method("selective_weight")
        S = torch.tensor([0.0, 1.0, 4.0, 4.0], dtype=torch.float64)
        V = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float64)
        B = torch.tensor([0.0, 0.0, 0.0, 100.0], dtype=torch.float64)

        actual = weight_fn(S, V, B, lmbda=0.5, epsilon=1e-6)

        # S=0 (no correction signal): weight collapses to ~0 regardless of V.
        self.assertAlmostEqual(actual[0].item(), 0.0, places=4)
        # V=0 (no finite-K uncertainty): weight goes to 1 (fully retain the correction).
        self.assertAlmostEqual(actual[1].item(), 1.0, places=4)
        # S and V balanced, no anchor support: matches S/(S+V).
        self.assertAlmostEqual(actual[2].item(), 4.0 / 5.0, places=4)
        # Large positive anchor-alignment term pushes the ratio above 1, so it clips to 1.
        self.assertAlmostEqual(actual[3].item(), 1.0, places=6)
        self.assertTrue(torch.all(actual >= 0.0) and torch.all(actual <= 1.0))

    def test_weighted_forward_kl_matches_manual_kl_times_weight(self):
        loss_fn = load_trainer_method("weighted_forward_kl_loss")
        student_logits = torch.tensor([[[0.2, -0.4, 0.7], [0.1, 0.1, 0.1]]], dtype=torch.float64)
        teacher_probs = torch.tensor(
            [[[0.6, 0.1, 0.3], [0.3, 0.3, 0.4]]], dtype=torch.float64
        )
        weights = torch.tensor([[2.0, 0.5]], dtype=torch.float64)
        labels = torch.tensor([[1, 1]])

        actual = loss_fn(student_logits, teacher_probs, weights, labels, temperature=1.0)

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        per_token_kl = (teacher_probs * (torch.log(teacher_probs) - student_log_probs)).sum(dim=-1)
        expected = (per_token_kl * weights).sum() / 2
        self.assertTrue(torch.allclose(actual, expected))

    def test_compute_loss_with_anchor_applies_selective_weight_and_updates_masking_store(self):
        compute_loss = load_trainer_method("compute_loss")
        selective_weight = load_trainer_method("selective_weight")
        weighted_forward_kl_loss = load_trainer_method("weighted_forward_kl_loss")

        student_target_logits = torch.tensor([0.1, 0.4, -0.2], dtype=torch.float64)
        view_logits = [
            torch.tensor([1.0, -0.2, 0.1], dtype=torch.float64),
            torch.tensor([-0.4, 1.2, 0.3], dtype=torch.float64),
        ]
        anchor_target_logits = torch.tensor([0.2, 0.1, 1.1], dtype=torch.float64)
        num_views = len(view_logits)

        class FakeModel:
            def __init__(self):
                self.calls = 0

            def __call__(self, input_ids, attention_mask):
                # Call order: student, K masked views, anchor, K masked views again
                # (adaptive-masking recompute pass).
                if self.calls == 0:
                    target = student_target_logits
                elif self.calls <= num_views:
                    target = view_logits[self.calls - 1]
                elif self.calls == num_views + 1:
                    target = anchor_target_logits
                else:
                    target = view_logits[self.calls - num_views - 2]
                self.calls += 1
                logits = torch.zeros((1, 3, 3), dtype=torch.float64)
                logits[:, 1, :] = target
                if self.calls == 1:
                    logits.requires_grad_()
                return SimpleNamespace(logits=logits)

        class FakeMaskStore:
            def __init__(self):
                self.updates = []

            def update(self, example_id, masks, utilities):
                self.updates.append((example_id, masks, utilities))

        mask_store = FakeMaskStore()
        trainer = SimpleNamespace(
            use_thinking_machines_loss=False,
            temperature=1.0,
            use_ema_teacher=False,
            fixed_teacher=False,
            beta=0,
            top_k_loss=None,
            jsd_token_clip=None,
            anchor_coefficient=0.5,
            weight_regularization=1e-6,
            mask_state_store=mask_store,
            generalized_jsd_loss=load_trainer_method("generalized_jsd_loss"),
            selective_weight=selective_weight,
            weighted_forward_kl_loss=weighted_forward_kl_loss,
        )
        trainer.compute_loss = MethodType(compute_loss, trainer)

        view_masks = torch.tensor([[True, False, True], [False, True, True]])
        inputs = {
            "student_prompt_length": 2,
            "teacher_prompt_length": 2,
            "anchor_prompt_length": 2,
            "student_input_ids": torch.tensor([[7, 8, 1]]),
            "student_attention_mask": torch.ones((1, 3), dtype=torch.long),
            "teacher_input_ids": torch.ones((num_views, 1, 3), dtype=torch.long),
            "teacher_attention_mask": torch.ones((num_views, 1, 3), dtype=torch.long),
            "anchor_prompts": torch.ones((1, 3), dtype=torch.long),
            "anchor_prompt_attention_mask": torch.ones((1, 3), dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 1]]),
            "view_masks": [view_masks],
            "example_ids": [42],
        }
        model = FakeModel()

        loss = trainer.compute_loss(model, inputs)

        self.assertTrue(torch.isfinite(loss))
        # student(1) + K masked views + anchor(1) + K masked views again (adaptive-masking recompute).
        self.assertEqual(model.calls, 2 * num_views + 2)

        self.assertEqual(len(mask_store.updates), 1)
        example_id, masks, utilities = mask_store.updates[0]
        self.assertEqual(example_id, 42)
        self.assertTrue(torch.equal(masks, view_masks))
        self.assertEqual(utilities.shape, (num_views,))
        self.assertTrue(torch.all(torch.isfinite(utilities)))

    def test_tinker_loss_logs_mean_sampled_probability(self):
        compute_loss = load_trainer_method("compute_loss")
        student_target_logits = torch.tensor([0.2, 0.8, -0.3], dtype=torch.float64)
        teacher_target_logits = [
            torch.tensor([1.1, 0.0, -0.5], dtype=torch.float64),
            torch.tensor([-0.2, 0.9, 0.4], dtype=torch.float64),
        ]

        class FakeModel:
            def __init__(self):
                self.calls = 0

            def __call__(self, input_ids, attention_mask):
                target = student_target_logits if self.calls == 0 else teacher_target_logits[self.calls - 1]
                self.calls += 1
                logits = torch.zeros((1, 3, 3), dtype=torch.float64)
                logits[:, 1, :] = target
                if self.calls == 1:
                    logits.requires_grad_()
                return SimpleNamespace(logits=logits)

        trainer = SimpleNamespace(
            use_thinking_machines_loss=True,
            temperature=1.0,
            use_ema_teacher=False,
            fixed_teacher=False,
        )
        trainer.compute_loss = MethodType(compute_loss, trainer)
        inputs = {
            "student_prompt_length": 2,
            "teacher_prompt_length": 2,
            "student_input_ids": torch.tensor([[7, 8, 1]]),
            "student_attention_mask": torch.ones((1, 3), dtype=torch.long),
            "teacher_input_ids": torch.ones((2, 1, 3), dtype=torch.long),
            "teacher_attention_mask": torch.ones((2, 1, 3), dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 1]]),
        }

        actual = trainer.compute_loss(FakeModel(), inputs)
        student_log_prob = F.log_softmax(student_target_logits, dim=-1)[1]
        mean_teacher_prob = torch.stack(
            [F.softmax(logits, dim=-1)[1] for logits in teacher_target_logits]
        ).mean()
        expected = -(torch.log(mean_teacher_prob) - student_log_prob).detach() * student_log_prob
        self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
