"""Hidden Markov Model for statistical regime classification.

Pure Python implementation — no numpy/scipy dependency.
Classification: FROZEN component.

4-state HMM:
  State 0: LOW_VOL_TRENDING
  State 1: HIGH_VOL_MEAN_REVERTING
  State 2: CRISIS_DISLOCATION
  State 3: TRANSITION_UNCERTAIN

The HMM uses a composite observation score derived from regime features.
Default parameters are calibrated for realistic market behavior. In
production, these would be fitted offline via the Learning subsystem.
"""

import math
from dataclasses import dataclass

from providence.agents.regime.regime_features import RegimeFeatures
from providence.schemas.enums import StatisticalRegime, SystemRiskMode

# State ordering (must match transition matrix indices)
REGIME_STATES = [
    StatisticalRegime.LOW_VOL_TRENDING,
    StatisticalRegime.HIGH_VOL_MEAN_REVERTING,
    StatisticalRegime.CRISIS_DISLOCATION,
    StatisticalRegime.TRANSITION_UNCERTAIN,
]
NUM_STATES = len(REGIME_STATES)


@dataclass(frozen=True)
class HMMParameters:
    """Parameters for the 4-state Hidden Markov Model.

    transition_matrix: 4x4 matrix (list of lists). [i][j] = P(state j | state i).
    emission_means: Mean of Gaussian emission for each state.
    emission_stds: Std dev of Gaussian emission for each state.
    initial_probs: Initial state distribution.
    """
    transition_matrix: tuple[tuple[float, ...], ...]
    emission_means: tuple[float, ...]
    emission_stds: tuple[float, ...]
    initial_probs: tuple[float, ...]


# Default calibrated parameters
# Composite observation: low values = calm markets, high values = stress
DEFAULT_HMM_PARAMS = HMMParameters(
    transition_matrix=(
        # From LOW_VOL_TRENDING:    stay=0.90, → HIGH_VOL=0.06, → CRISIS=0.01, → TRANSITION=0.03
        (0.90, 0.06, 0.01, 0.03),
        # From HIGH_VOL_MEAN_REV:   → LOW_VOL=0.10, stay=0.75, → CRISIS=0.05, → TRANSITION=0.10
        (0.10, 0.75, 0.05, 0.10),
        # From CRISIS_DISLOCATION:  → LOW_VOL=0.02, → HIGH_VOL=0.15, stay=0.70, → TRANSITION=0.13
        (0.02, 0.15, 0.70, 0.13),
        # From TRANSITION_UNCERTAIN: → LOW_VOL=0.20, → HIGH_VOL=0.30, → CRISIS=0.10, stay=0.40
        (0.20, 0.30, 0.10, 0.40),
    ),
    # Emission means: composite stress score
    # LOW_VOL=0.2, HIGH_VOL=0.5, CRISIS=0.85, TRANSITION=0.45
    emission_means=(0.20, 0.50, 0.85, 0.45),
    # Emission std devs
    emission_stds=(0.12, 0.15, 0.10, 0.20),
    # Initial state distribution
    initial_probs=(0.50, 0.25, 0.05, 0.20),
)


def gaussian_emission_prob(x: float, mean: float, std: float) -> float:
    """Compute Gaussian emission probability P(x | state).

    Args:
        x: Observation value.
        mean: Mean for this state's emission distribution.
        std: Standard deviation for this state's emission distribution.

    Returns:
        Probability density value (unnormalized for HMM purposes).
    """
    if std <= 0:
        return 1.0 if abs(x - mean) < 1e-10 else 1e-300

    exponent = -0.5 * ((x - mean) / std) ** 2
    normalizer = 1.0 / (std * math.sqrt(2 * math.pi))
    return normalizer * math.exp(exponent)


def features_to_composite_score(features: RegimeFeatures) -> float:
    """Convert regime features into a single composite stress score [0, 1].

    Higher values indicate more market stress. Uses a weighted combination
    of available features, normalized to [0, 1].

    Args:
        features: Extracted regime features.

    Returns:
        Composite stress score between 0.0 and 1.0.
    """
    scores: list[float] = []
    weights: list[float] = []

    # Realized volatility (higher vol = more stress)
    # Typical range: 0.08 (calm) to 0.80 (crisis)
    if features.realized_vol_20d is not None:
        vol_score = min(1.0, max(0.0, (features.realized_vol_20d - 0.08) / 0.72))
        scores.append(vol_score)
        weights.append(0.30)

    # Vol-of-vol (higher = more unstable)
    # Typical range: 0.02 (stable) to 0.40 (chaotic)
    if features.vol_of_vol is not None:
        vov_score = min(1.0, max(0.0, (features.vol_of_vol - 0.02) / 0.38))
        scores.append(vov_score)
        weights.append(0.10)

    # Yield curve inversion (more negative = more stress)
    # Range: +200bps (healthy) to -100bps (deeply inverted)
    if features.yield_spread_2s10s is not None:
        # Inversion is stress: normalize so -100bps → 1.0, +200bps → 0.0
        yield_score = min(1.0, max(0.0, (200.0 - features.yield_spread_2s10s) / 300.0))
        scores.append(yield_score)
        weights.append(0.15)

    # CDS spread (wider = more stress)
    # Typical range: 50bps (healthy) to 500bps (crisis)
    if features.cds_ig_spread is not None:
        cds_score = min(1.0, max(0.0, (features.cds_ig_spread - 50.0) / 450.0))
        scores.append(cds_score)
        weights.append(0.15)

    # VIX proxy (higher implied vol = more stress)
    # Range: 0.10 (calm) to 0.60 (panic)
    if features.vix_proxy is not None:
        vix_score = min(1.0, max(0.0, (features.vix_proxy - 0.10) / 0.50))
        scores.append(vix_score)
        weights.append(0.20)

    # Drawdown (more negative = more stress)
    # Range: 0% (no drawdown) to -30% (severe)
    if features.price_drawdown_pct is not None:
        dd_score = min(1.0, max(0.0, abs(features.price_drawdown_pct) / 0.30))
        scores.append(dd_score)
        weights.append(0.10)

    if not scores:
        # No features available — return neutral
        return 0.45

    # Weighted average
    total_weight = sum(weights)
    composite = sum(s * w for s, w in zip(scores, weights)) / total_weight
    return min(1.0, max(0.0, composite))


def forward_algorithm(
    observation: float,
    params: HMMParameters = DEFAULT_HMM_PARAMS,
    prior: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    """Run one step of the HMM forward algorithm.

    Given a single observation and prior state distribution, compute
    the posterior state probabilities.

    Args:
        observation: Composite stress score.
        params: HMM parameters.
        prior: Prior state distribution. If None, uses initial_probs.

    Returns:
        Tuple of posterior state probabilities (one per state).
    """
    if prior is None:
        prior = params.initial_probs

    # Predict step: apply transition matrix
    predicted = [0.0] * NUM_STATES
    for j in range(NUM_STATES):
        for i in range(NUM_STATES):
            predicted[j] += prior[i] * params.transition_matrix[i][j]

    # Update step: multiply by emission probabilities
    emission_probs = [
        gaussian_emission_prob(observation, params.emission_means[j], params.emission_stds[j])
        for j in range(NUM_STATES)
    ]

    posterior = [predicted[j] * emission_probs[j] for j in range(NUM_STATES)]

    # Normalize
    total = sum(posterior)
    if total > 0:
        posterior = [p / total for p in posterior]
    else:
        # Fallback to uniform
        posterior = [1.0 / NUM_STATES] * NUM_STATES

    return tuple(posterior)


def classify_regime(
    features: RegimeFeatures,
    params: HMMParameters = DEFAULT_HMM_PARAMS,
    prior: tuple[float, ...] | None = None,
) -> tuple[StatisticalRegime, float, dict[str, float]]:
    """Classify the current market regime from features.

    Args:
        features: Extracted regime features.
        params: HMM parameters.
        prior: Prior state distribution (from previous classification).

    Returns:
        Tuple of (classified regime, confidence, probability dict).
    """
    observation = features_to_composite_score(features)
    posteriors = forward_algorithm(observation, params, prior)

    # Find the most likely state
    max_idx = 0
    max_prob = posteriors[0]
    for i in range(1, NUM_STATES):
        if posteriors[i] > max_prob:
            max_prob = posteriors[i]
            max_idx = i

    regime = REGIME_STATES[max_idx]
    confidence = max_prob

    # Build probability dict
    prob_dict = {
        state.value: round(prob, 6) for state, prob in zip(REGIME_STATES, posteriors)
    }

    return regime, confidence, prob_dict


def derive_risk_mode(
    regime: StatisticalRegime,
    confidence: float,
) -> SystemRiskMode:
    """Derive system risk mode from regime classification.

    Mapping:
      LOW_VOL_TRENDING       → NORMAL
      HIGH_VOL_MEAN_REVERTING → CAUTIOUS
      CRISIS_DISLOCATION     → DEFENSIVE (HALTED if confidence > 0.9)
      TRANSITION_UNCERTAIN   → CAUTIOUS

    Args:
        regime: Classified regime state.
        confidence: Confidence in the classification.

    Returns:
        SystemRiskMode governing position sizing and exposure limits.
    """
    if regime == StatisticalRegime.LOW_VOL_TRENDING:
        return SystemRiskMode.NORMAL
    elif regime == StatisticalRegime.HIGH_VOL_MEAN_REVERTING:
        return SystemRiskMode.CAUTIOUS
    elif regime == StatisticalRegime.CRISIS_DISLOCATION:
        if confidence > 0.9:
            return SystemRiskMode.HALTED
        return SystemRiskMode.DEFENSIVE
    else:  # TRANSITION_UNCERTAIN
        return SystemRiskMode.CAUTIOUS
