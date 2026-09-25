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
