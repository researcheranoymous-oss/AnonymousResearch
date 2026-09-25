import re

import torch


_LATEX_PATTERN = re.compile(
    r"(?P<display>\$\$.*?\$\$)"
    r"|(?P<inline>(?<!\\)\$(?!\$).*?(?<!\\)\$)"
    r"|(?P<paren>\\\(.*?\\\))"
    r"|(?P<bracket>\\\[.*?\\\])"
    r"|(?P<environment>\\begin\{(?P<environment_name>[^{}]+)\}.*?"
    r"\\end\{(?P=environment_name)\})",
    flags=re.DOTALL,
)


def _split_plain_text(text):
    """Split prose after line breaks and sentence-ending punctuation."""
    spans = []
    start = 0
    for index, character in enumerate(text):
        is_line_break = character == "\n"
        is_sentence_end = character in ".!?;。！？；" and (
            index + 1 == len(text) or text[index + 1].isspace()
        )
        if is_line_break or is_sentence_end:
            spans.append(text[start : index + 1])
            start = index + 1
    if start < len(text):
        spans.append(text[start:])
    return spans


def split_reference_into_spans(reference):
    """Split a reference while keeping complete LaTeX expressions as atomic spans."""
    spans = []
    cursor = 0
    for match in _LATEX_PATTERN.finditer(reference):
        spans.extend(_split_plain_text(reference[cursor : match.start()]))
        spans.append(match.group(0))
        cursor = match.end()
    spans.extend(_split_plain_text(reference[cursor:]))
    spans = [span for span in spans if span]

    # A one-span reference cannot satisfy the invariant that every view contains
    # both visible and unavailable content. Split it near its midpoint.
    if len([span for span in spans if not span.isspace()]) < 2 and reference:
        midpoint = len(reference) // 2
        whitespace_positions = [index for index, char in enumerate(reference) if char.isspace()]
        split_at = min(whitespace_positions, key=lambda index: abs(index - midpoint), default=midpoint)
        split_at = max(1, min(len(reference) - 1, split_at + 1))
        spans = [reference[:split_at], reference[split_at:]]

    return spans


def mask_reference(reference, probability):
    """Sample one masked view, preserving character count and whitespace."""
    spans = split_reference_into_spans(reference)
    maskable_indices = [index for index, span in enumerate(spans) if not span.isspace()]
    if len(maskable_indices) < 2:
        return reference

    for _ in range(1024):
        decisions = torch.rand(len(maskable_indices)) < probability
        if decisions.any() and not decisions.all():
            break
    else:
        # Avoid an infinite loop for probabilities that round to 0 or 1 in
        # torch.rand's dtype while preserving the visible/masked invariant.
        decisions = torch.rand(len(maskable_indices)) < probability
        forced = torch.randperm(len(maskable_indices))[:2]
        decisions[forced[0]] = True
        decisions[forced[1]] = False

    masked_indices = {
        span_index for span_index, should_mask in zip(maskable_indices, decisions.tolist()) if should_mask
    }
    return "".join(
        "".join(char if char.isspace() else "#" for char in span) if index in masked_indices else span
        for index, span in enumerate(spans)
    )


class SelfDistillationDataCollator:
    """
    Data collator for self-distillation that creates both student and teacher inputs.

    Student: sees only the problem (with chat template)
    Teacher: sees problem + solution + transition prompt (with chat template)

    To enable batch-level operations (like original GKD), we pad prompts to the same length
    within each batch, and track the actual (unpadded) prompt lengths for loss masking.
    """

    def __init__(
        self,
        tokenizer,
        max_length=2048,
        reason_first=False,
        student_thinking=False,
        teacher_thinking=True,
        use_reference_masking=True,
        reference_mask_probability=0.35,
        num_masked_teacher_views=8,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reason_first = reason_first
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking
        self.use_reference_masking = use_reference_masking
        self.reference_mask_probability = reference_mask_probability
        self.num_masked_teacher_views = num_masked_teacher_views if use_reference_masking else 1

        if self.use_reference_masking and not 0 < self.reference_mask_probability < 1:
            raise ValueError("reference_mask_probability must be strictly between 0 and 1.")
        if self.use_reference_masking and self.num_masked_teacher_views < 1:
            raise ValueError("num_masked_teacher_views must be at least 1.")
        if self.use_reference_masking and self.reason_first:
            raise ValueError("Masked-reference training does not support reason_first=True.")

        # Prompt for reasoning about the solution before teaching
        self.reason_first_prompt = (
            "\n\nThe reference reasoning above arrives at the correct answer. "
            "Please analyze this solution and explain the key reasoning steps and problem-solving strategies employed. "
            "Do NOT use <think> tags. Do NOT derive your own solution. "
            "Simply analyze and explain the reference solution provided above.\n"
        )
        # Prompt for transitioning to teaching mode after reasoning
        self.transition_prompt = (
            "\n\nAfter reading the reference solution above, make sure you truly understand "
            "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
            "own words and independent reasoning, derive the same final answer to the problem above. "
            "Think step by step, explore different approaches, and don't be afraid to backtrack "
            "or reconsider if something doesn't work out:\n"
        )
        self.reference_availability_prompt = (
            "\nSome parts of the reference may appear as # placeholders and are unavailable. "
            "Use the visible context to understand the approach and reason about the solution.\n"
        )

        # Set padding side explicitly for consistency
        print(f"[DataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[DataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[DataCollator] Reason first mode: {self.reason_first}")
        print(
            "[DataCollator] Reference mode: "
            + (
                f"masked ({self.num_masked_teacher_views} views, p={self.reference_mask_probability})"
                if self.use_reference_masking
                else "full reference (1 view)"
            )
        )

    def __call__(self, features):

        batch_size = len(features)

        # Prepare student and teacher prompts using chat template (matching evaluation)
        student_prompts = []
        teacher_prompts_by_view = [[] for _ in range(self.num_masked_teacher_views)]
        teacher_reasoning_prompts = []  # NEW: for reason_first mode

        for feature in features:
            # Extract problem and solution from dataset
            # Handle different possible column names
            problem = feature["problem"]
            solution = feature["solution"]

            # Student prompt: just the problem with instruction (matching evaluation format)
            student_user_message = f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            student_messages = [{"role": "user", "content": student_user_message}]

            # Apply chat template for student (matching evaluation)
            student_prompt = self.tokenizer.apply_chat_template(
                student_messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.student_thinking
            )
            student_prompts.append(student_prompt)

            if self.reason_first:
                # Reasoning prompt: ask teacher to analyze the solution
                reasoning_user_message = (
                    f"Problem: {problem}\n\n"
                    f"Here is a correct reasoning to this problem:"
                    f"=== Reference Reasoning Start ===\n"
                    f"{solution}\n"
                    f"=== Reference Reasoning End ===\n\n"
                    f"{self.reason_first_prompt}"
                )
                reasoning_messages = [{"role": "user", "content": reasoning_user_message}]
                reasoning_prompt = self.tokenizer.apply_chat_template(
                    reasoning_messages, tokenize=False, add_generation_prompt=True
                )
                teacher_reasoning_prompts.append(reasoning_prompt)

                # Teacher prompt will be constructed during training after reasoning
                # For now, create placeholder (will be replaced in training_step)
                # This branch is rejected in __init__; retained for structural clarity.
            else:
                for view_index in range(self.num_masked_teacher_views):
                    teacher_solution = (
                        mask_reference(solution, self.reference_mask_probability)
                        if self.use_reference_masking
                        else solution
                    )
                    teacher_user_message = (
                        f"Problem: {problem}\n\n"
                        f"Here is a reference solution to this problem:\n"
                        f"=== Reference Solution Begin ===\n{teacher_solution}\n=== Reference Solution End ===\n"
                        f"{self.reference_availability_prompt if self.use_reference_masking else ''}"
                        f"{self.transition_prompt}\n"
                        f"Please reason step by step, and put your final answer within \\boxed{{}}."
                    )
                    teacher_messages = [{"role": "user", "content": teacher_user_message}]
                    teacher_prompt = self.tokenizer.apply_chat_template(
                        teacher_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=self.teacher_thinking,
                    )
                    teacher_prompts_by_view[view_index].append(teacher_prompt)

        # Tokenize WITHOUT padding first to get true lengths
        student_encoded_no_pad = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]

        # Find max lengths in this batch
        max_student_prompt_len = max(student_prompt_lengths)

        # Tokenize WITH padding to max length in batch
        student_encoded = self.tokenizer(
            student_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_student_prompt_len,
            return_tensors="pt",
        )

        result = {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,  # Single value for batch!
            # Keep individual lengths for proper masking
            "student_prompt_lengths_per_example": torch.tensor(student_prompt_lengths),
        }

        if self.reason_first:
            # Tokenize reasoning prompts
            reasoning_encoded_no_pad = self.tokenizer(
                teacher_reasoning_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            reasoning_prompt_lengths = [len(ids) for ids in reasoning_encoded_no_pad["input_ids"]]
            max_reasoning_prompt_len = max(reasoning_prompt_lengths)

            reasoning_encoded = self.tokenizer(
                teacher_reasoning_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_reasoning_prompt_len,
                return_tensors="pt",
            )

            # Tokenize transition prompt (this will be appended after reasoning)
            # Don't use chat template here - just the raw text
            transition_text = f"\n{self.transition_prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            transition_encoded = self.tokenizer(
                [transition_text] * batch_size,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_reasoning_prompts": reasoning_encoded["input_ids"],
                    "teacher_reasoning_attention_mask": reasoning_encoded["attention_mask"],
                    "teacher_reasoning_prompt_length": max_reasoning_prompt_len,
                    "teacher_transition_tokens": transition_encoded["input_ids"],
                }
            )
        else:
            # Flatten in view-major order, tokenize once, then restore [views, batch, sequence].
            teacher_prompts = [prompt for view in teacher_prompts_by_view for prompt in view]
            teacher_encoded_no_pad = self.tokenizer(
                teacher_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            teacher_prompt_lengths = [len(ids) for ids in teacher_encoded_no_pad["input_ids"]]
            max_teacher_prompt_len = max(teacher_prompt_lengths)

            teacher_encoded = self.tokenizer(
                teacher_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_teacher_prompt_len,
                return_tensors="pt",
            )

            # Teacher prompts are followed by the same generation. Left padding keeps
            # every prompt's final real token immediately before that generation, so the
            # first teacher target never comes from a right-padding position.
            teacher_input_ids = torch.full_like(
                teacher_encoded["input_ids"], self.tokenizer.pad_token_id
            )
            teacher_attention_mask = torch.zeros_like(teacher_encoded["attention_mask"])
            for row_index, prompt_length in enumerate(teacher_prompt_lengths):
                teacher_input_ids[row_index, -prompt_length:] = teacher_encoded["input_ids"][
                    row_index, :prompt_length
                ]
                teacher_attention_mask[row_index, -prompt_length:] = 1

            result.update(
                {
                    "teacher_prompts": teacher_input_ids.reshape(
                        self.num_masked_teacher_views, batch_size, max_teacher_prompt_len
                    ),
                    "teacher_prompt_attention_mask": teacher_attention_mask.reshape(
                        self.num_masked_teacher_views, batch_size, max_teacher_prompt_len
                    ),
                    "teacher_prompt_length": max_teacher_prompt_len,
                    "teacher_prompt_lengths_per_example": torch.tensor(teacher_prompt_lengths).reshape(
                        self.num_masked_teacher_views, batch_size
                    ),
                }
            )

        return result
