import unittest

import torch

from data_collator import SelfDistillationDataCollator, mask_reference, split_reference_into_spans


class FakeTokenizer:
    pad_token_id = 0
    padding_side = "left"

    def __init__(self):
        self.chat_contents = []

    def apply_chat_template(self, messages, **kwargs):
        content = messages[0]["content"]
        self.chat_contents.append(content)
        return content + "<assistant>"

    def __call__(
        self,
        texts,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
    ):
        encoded = [[(ord(char) % 250) + 1 for char in text] for text in texts]
        if truncation and max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]

        if padding == "max_length":
            encoded = [ids + [self.pad_token_id] * (max_length - len(ids)) for ids in encoded]

        masks = [[int(token != self.pad_token_id) for token in ids] for ids in encoded]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(encoded), "attention_mask": torch.tensor(masks)}
        return {"input_ids": encoded, "attention_mask": masks}


class ReferenceMaskingTest(unittest.TestCase):
    def test_split_preserves_text_and_latex_units(self):
        reference = (
            "First sentence. Inline \\(x+1.5=y\\) stays whole.\n"
            "Display: \\[a.b=c\\]\n"
            "Dollar $u.v=w$ and $$q.r=s$$.\n"
            "\\begin{align}x &= 1.2 \\\\ y &= 3\\end{align}\nDone!"
        )
        spans = split_reference_into_spans(reference)

        self.assertEqual("".join(spans), reference)
        for formula in (
            "\\(x+1.5=y\\)",
            "\\[a.b=c\\]",
            "$u.v=w$",
            "$$q.r=s$$",
            "\\begin{align}x &= 1.2 \\\\ y &= 3\\end{align}",
        ):
            self.assertIn(formula, spans)

    def test_mask_preserves_length_and_whitespace(self):
        reference = "One sentence. Two sentence! \\[x+y=z\\]\nFinal sentence."
        torch.manual_seed(7)
        masked = mask_reference(reference, 0.35)

        self.assertEqual(len(masked), len(reference))
        self.assertIn("#", masked)
        self.assertTrue(any(a == b and not a.isspace() for a, b in zip(reference, masked)))
        self.assertEqual(
            [index for index, char in enumerate(reference) if char.isspace()],
            [index for index, char in enumerate(masked) if char.isspace()],
        )

    def test_collator_returns_view_major_teacher_tensors(self):
        tokenizer = FakeTokenizer()
        collator = SelfDistillationDataCollator(
            tokenizer,
            max_length=4096,
            reason_first=False,
            reference_mask_probability=0.35,
            num_masked_teacher_views=3,
        )
        torch.manual_seed(11)
        batch = collator(
            [
                {"problem": "P1", "solution": "First. Second. \\[x=1\\] Third."},
                {"problem": "P2", "solution": "Alpha. Beta. $y=2$ Gamma."},
            ]
        )

        self.assertEqual(batch["teacher_prompts"].shape[:2], (3, 2))
        self.assertEqual(batch["teacher_prompt_attention_mask"].shape, batch["teacher_prompts"].shape)
        self.assertEqual(batch["teacher_prompt_lengths_per_example"].shape, (3, 2))
        self.assertTrue(torch.all(batch["teacher_prompt_attention_mask"][:, :, -1] == 1))
        self.assertTrue(torch.any(batch["teacher_prompt_attention_mask"][:, :, 0] == 0))
        teacher_contents = [
            prompt for prompt in tokenizer.chat_contents if "Here is a reference solution" in prompt
        ]
        self.assertEqual(len(teacher_contents), 6)
        self.assertTrue(all("# placeholders" in prompt for prompt in teacher_contents))
        self.assertTrue(all("#" in prompt for prompt in teacher_contents))

    def test_invalid_configuration_is_rejected(self):
        tokenizer = FakeTokenizer()
        with self.assertRaisesRegex(ValueError, "strictly between"):
            SelfDistillationDataCollator(tokenizer, reference_mask_probability=0)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            SelfDistillationDataCollator(tokenizer, num_masked_teacher_views=0)
        with self.assertRaisesRegex(ValueError, "reason_first"):
            SelfDistillationDataCollator(tokenizer, reason_first=True)

    def test_masking_disabled_restores_single_full_reference(self):
        tokenizer = FakeTokenizer()
        collator = SelfDistillationDataCollator(
            tokenizer,
            max_length=4096,
            use_reference_masking=False,
            reference_mask_probability=0,
            num_masked_teacher_views=0,
        )
        solution = "First complete step. \\[x=1\\] Final complete step."
        batch = collator([{"problem": "P", "solution": solution}])
        teacher_prompt = next(
            prompt for prompt in tokenizer.chat_contents if "Here is a reference solution" in prompt
        )

        self.assertEqual(batch["teacher_prompts"].shape[0], 1)
        self.assertIn(solution, teacher_prompt)
        self.assertNotIn("# placeholders", teacher_prompt)


if __name__ == "__main__":
    unittest.main()
