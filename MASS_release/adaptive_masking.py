"""Adaptive per-span masking: Shapley-kernel attribution and the closed-form
masking-probability update described in the MASS paper (Sec. 2.3, Eqs. 7-10,
Appendix A.5-A.6).

NOTE ON THIS RELEASE: the adaptive-masking internals (Shapley-kernel regression
and the closed-form masking-probability update) are withheld from this
anonymous review bundle and will be released upon paper acceptance. This
module still implements plain per-span uniform Bernoulli masking (Sec. 2.1,
Eq. 1) in full, and ``AdaptiveMaskingStore`` defaults to a non-adaptive
no-op mode so the rest of the codebase (masked-average teacher, anchor
teacher, selective weighting, Sec. 2.1-2.2) runs end to end without it. This
reproduces the paper's disclosed "Weighting only" ablation row (Table 3),
not the "Full MASS" row, which additionally requires the withheld component.

Kept as a standalone module (no dependency on the trainer or data collator)
so it can be unit tested in isolation and imported from both.
"""

import torch


class AdaptiveMaskingWithheld(NotImplementedError):
    """Raised when adaptive Shapley masking is requested but not included in
    this release. The internals will be published upon paper acceptance.
    """


class SpanMaskState:
    """Per-example adaptive-masking state: current per-span masking
    probabilities ``q`` and the EMA-smoothed Shapley attribution ``z``.
    """

    __slots__ = ("q", "z")

    def __init__(self, num_spans, rho):
        self.q = torch.full((num_spans,), float(rho), dtype=torch.float64)
        self.z = torch.zeros(num_spans, dtype=torch.float64)


class AdaptiveMaskingStore:
    """Rank-local (no cross-process sync) map from example id to its
    :class:`SpanMaskState`. Each GPU process keeps state only for the
    examples it happens to process; this is a deliberate simplification for
    single-node multi-GPU training (see plan for rationale).
    """

    def __init__(
        self,
        rho,
        mask_probability_lower_bound,
        mask_adaptation_rate,
        ema_smoothing,
        shapley_regularization,
        adaptive=False,
    ):
        self.rho = rho
        self.epsilon = mask_probability_lower_bound
        self.eta = mask_adaptation_rate
        self.gamma_ema = ema_smoothing
        self.tau = shapley_regularization
        # Withheld in this release (see module docstring). When False (default),
        # update() is a no-op: q stays fixed at rho, i.e. plain uniform masking.
        self.adaptive = adaptive
        self._states = {}

    def get_or_init(self, example_id, num_spans):
        state = self._states.get(example_id)
        if state is None or state.q.numel() != num_spans:
            state = SpanMaskState(num_spans, self.rho)
            self._states[example_id] = state
        return state

    def update(self, example_id, masks, utilities):
        """Run the Shapley regression + masking-probability update for one example.

        Args:
            example_id: key into the store (must already exist via ``get_or_init``).
            masks: bool/float tensor [K, J] of the K sampled span-mask coalitions
                used for this example's distillation this step.
            utilities: tensor [K] with the per-view coalition utility v(A_k)
                (Eq. 7), computed by the caller from cached teacher/student probs.
        """
        if not self.adaptive:
            return
        state = self._states[example_id]
        phi_hat = shapley_regression(masks, utilities, state.z, tau=self.tau)
        update_span_state(state, phi_hat, self.rho, self.epsilon, self.eta, self.gamma_ema)


def sample_bernoulli_mask(q, generator=None):
    """Sample one mask from independent Bernoulli(q_j), retrying to avoid the
    degenerate all-visible / all-masked draw (mirrors the original invariant).
    """
    num_spans = q.numel()
    if num_spans < 2:
        return torch.zeros(num_spans, dtype=torch.bool)
    for _ in range(1024):
        draw = torch.rand(num_spans, generator=generator) < q
        if draw.any() and not draw.all():
            return draw
    draw = torch.rand(num_spans, generator=generator) < q
    forced = torch.randperm(num_spans)[:2]
    draw[forced[0]] = True
    draw[forced[1]] = False
    return draw


def sample_k_masks(q, k, generator=None):
    """Draw K independent masks from Bernoulli(q) (Sec. 2.1: m^(k) ~ q)."""
    return torch.stack([sample_bernoulli_mask(q, generator=generator) for _ in range(k)])


def shapley_regression(masks, utilities, z_prior, tau=0.1):
    """Withheld in this release.

    In the full implementation, this solves the ridge-regularized
    Shapley-kernel weighted least squares of Eq. 9 (closed-form KKT system,
    Appendix A.5) to attribute each reference span's causal effect on the
    mask-marginal teacher correction, using the K sampled coalitions already
    evaluated for distillation. Will be released upon paper acceptance.
    """
    raise AdaptiveMaskingWithheld(
        "adaptive_masking.shapley_regression is withheld pending paper acceptance. "
        "Construct AdaptiveMaskingStore with adaptive=False (the default) to run "
        "MASS with plain uniform per-span masking instead."
    )


def update_span_state(state, phi_hat, rho, epsilon, eta, gamma_ema):
    """Withheld in this release.

    In the full implementation, this EMA-smooths the Shapley attribution
    estimate and applies the closed-form masking-probability update (Eq. 10 /
    Prop. A.10). Will be released upon paper acceptance.
    """
    raise AdaptiveMaskingWithheld(
        "adaptive_masking.update_span_state is withheld pending paper acceptance. "
        "Construct AdaptiveMaskingStore with adaptive=False (the default) to run "
        "MASS with plain uniform per-span masking instead."
    )
