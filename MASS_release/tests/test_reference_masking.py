import unittest

import torch

from data_collator import (
    SelfDistillationDataCollator,
    mask_reference,
    maskable_span_indices,
    split_reference_into_spans,
)


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

    def test_mask_replaces_whole_spans_with_mask_token(self):
        reference = "One sentence. Two sentence! \\[x+y=z\\]\nFinal sentence."
        spans = split_reference_into_spans(reference)
        maskable = maskable_span_indices(spans)
        torch.manual_seed(7)
        masked = mask_reference(reference, 0.35)

        self.assertIn("[MASK]", masked)
        # Reconstruct which spans were masked from the rendered text: a masked
        # span disappears in favor of one literal "[MASK]" token, so every
        # OTHER maskable span's exact text must still appear verbatim.
        stripped = masked.replace("[MASK]", "")
        at_least_one_masked = any(spans[index] not in masked for index in maskable)
        at_least_one_visible = any(spans[index] in stripped for index in maskable)
        self.assertTrue(at_least_one_masked)
        self.assertTrue(at_least_one_visible)

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
        # 3 masked views + 1 anchor (full-reference) prompt per example.
        self.assertEqual(len(teacher_contents), 8)

        view_contents = [prompt for prompt in teacher_contents if "[MASK] placeholders" in prompt]
        anchor_contents = [prompt for prompt in teacher_contents if "[MASK] placeholders" not in prompt]
        self.assertEqual(len(view_contents), 6)
        self.assertEqual(len(anchor_contents), 2)
        self.assertTrue(all("[MASK]" in prompt for prompt in view_contents))
        self.assertTrue(all("[MASK]" not in prompt for prompt in anchor_contents))

        self.assertEqual(len(batch["view_masks"]), 2)
        for masks in batch["view_masks"]:
            self.assertEqual(masks.shape[0], 3)  # K views
        self.assertEqual(batch["example_ids"], [-1, -1])
        self.assertEqual(batch["anchor_prompts"].shape[0], 2)

    def test_invalid_configuration_is_rejected(self):
        tokenizer = FakeTokenizer()
        with self.assertRaisesRegex(ValueError, "strictly between"):
            SelfDistillationDataCollator(tokenizer, reference_mask_probability=0)
        with self.assertRaisesRegex(ValueError, "at least 2"):
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
