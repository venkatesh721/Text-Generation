"""Temperature-based next-word sampling and seed-based text generation."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.preprocessing import Vocabulary, tokenize_text


def generate_text(
    model: Any,
    vocabulary: Vocabulary,
    seed_text: str,
    sequence_length: int,
    num_tokens: int,
    temperature: float,
    rng: np.random.Generator,
) -> str:
    """Generate words from a seed, mapping unseen seed words to <unk>."""
    if sequence_length < 1:
        raise ValueError("Sequence length must be at least 1.")
    if num_tokens < 0:
        raise ValueError("Number of generated tokens cannot be negative.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be a finite number greater than 0.")

    seed_tokens = tokenize_text(seed_text)
    if not seed_tokens:
        raise ValueError("Seed text must contain at least one word.")
    context = vocabulary.encode(seed_tokens).tolist()

    for _ in range(num_tokens):
        window = context[-sequence_length:]
        if len(window) < sequence_length:
            window = [0] * (sequence_length - len(window)) + window
        predictions = np.asarray(
            model.predict(np.asarray([window], dtype=np.int32), verbose=0),
            dtype=np.float64,
        )
        if predictions.shape != (1, vocabulary.size):
            raise ValueError("Model output size does not match the vocabulary.")
        probabilities = predictions[0]
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError("Model returned invalid token probabilities.")
        probability_sum = float(probabilities.sum())
        if not np.isfinite(probability_sum) or probability_sum <= 0:
            raise ValueError("Model returned token probabilities with no positive mass.")

        probabilities = probabilities / probability_sum
        logits = np.full(probabilities.shape, -np.inf, dtype=np.float64)
        positive = probabilities > 0
        logits[positive] = np.log(probabilities[positive]) / temperature
        logits -= np.max(logits)
        sampling_probabilities = np.exp(logits)
        sampling_total = sampling_probabilities.sum()
        if not np.isfinite(sampling_total) or sampling_total <= 0:
            raise ValueError("Could not normalize model predictions for sampling.")
        sampling_probabilities /= sampling_total
        next_token = int(rng.choice(vocabulary.size, p=sampling_probabilities))
        context.append(next_token)

    return " ".join(vocabulary.decode(context))
