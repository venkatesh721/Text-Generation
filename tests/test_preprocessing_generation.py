"""Focused tests for cleaning, chronological splitting, and text generation."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.generation import generate_text
from src.preprocessing import (
    Vocabulary,
    clean_text,
    download_dataset,
    prepare_train_validation,
)


class FixedModel:
    """Return a fixed next-token distribution without requiring TensorFlow."""

    def __init__(self, vocabulary_size: int, selected_token: int) -> None:
        self.vocabulary_size = vocabulary_size
        self.selected_token = selected_token

    def predict(self, inputs: np.ndarray, verbose: int = 0) -> np.ndarray:
        self.last_inputs = inputs
        probabilities = np.zeros((1, self.vocabulary_size))
        probabilities[0, self.selected_token] = 1.0
        return probabilities


class PreprocessingAndGenerationTests(unittest.TestCase):
    def test_clean_text_removes_gutenberg_boilerplate_and_punctuation(self) -> None:
        text = (
            "License details\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            "Hello, WORLD!\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            "Footer"
        )

        self.assertEqual(clean_text(text), "\nhello  world \n")

    def test_validation_words_do_not_enter_training_vocabulary(self) -> None:
        tokens = "a b c d e f g h i j".split()

        vocabulary, train_x, train_y, validation_x, validation_y = (
            prepare_train_validation(tokens, 2, 0.3, None, None)
        )

        self.assertEqual(len(train_x), 5)
        self.assertEqual(len(train_y), 5)
        self.assertEqual(len(validation_x), 3)
        self.assertTrue(np.all(validation_y == 0))
        self.assertEqual(validation_x[0].tolist(), vocabulary.encode(["f", "g"]).tolist())
        self.assertEqual(vocabulary.encode(["h"])[0], 0)

    def test_generation_maps_unknown_seed_words_to_unknown_token(self) -> None:
        vocabulary = Vocabulary.from_tokens(["a", "b"])
        model = FixedModel(vocabulary.size, vocabulary.word_to_id["a"])

        result = generate_text(
            model,
            vocabulary,
            "Unseen seed",
            sequence_length=2,
            num_tokens=2,
            temperature=0.8,
            rng=np.random.default_rng(42),
        )

        self.assertEqual(result, "<unk> <unk> a a")

    def test_generation_rejects_seed_without_words(self) -> None:
        vocabulary = Vocabulary.from_tokens(["word"])
        model = FixedModel(vocabulary.size, vocabulary.word_to_id["word"])

        with self.assertRaisesRegex(ValueError, "at least one word"):
            generate_text(
                model,
                vocabulary,
                "!!!",
                sequence_length=2,
                num_tokens=1,
                temperature=1.0,
                rng=np.random.default_rng(42),
            )

    def test_sequences_align_context_with_next_token(self) -> None:
        tokens = ["one", "two", "three", "four", "five", "six", "seven", "eight"]

        vocabulary, train_x, train_y, validation_x, validation_y = (
            prepare_train_validation(tokens, 2, 0.25, None, None)
        )

        self.assertEqual(
            vocabulary.decode(train_x[0]), ["one", "two"]
        )
        self.assertEqual(vocabulary.decode([train_y[0]]), ["three"])
        self.assertEqual(
            vocabulary.decode(validation_x[0]), ["five", "six"]
        )
        self.assertEqual(vocabulary.decode([validation_y[0]]), ["<unk>"])

    def test_generation_rejects_invalid_probability_mass(self) -> None:
        vocabulary = Vocabulary.from_tokens(["a", "b"])

        class EmptyDistribution:
            def predict(self, inputs: np.ndarray, verbose: int = 0) -> np.ndarray:
                return np.zeros((1, vocabulary.size))

        with self.assertRaisesRegex(ValueError, "no positive mass"):
            generate_text(
                EmptyDistribution(),
                vocabulary,
                "a",
                sequence_length=1,
                num_tokens=1,
                temperature=1.0,
                rng=np.random.default_rng(42),
            )

    def test_download_does_not_overwrite_existing_dataset_by_default(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "shakespeare.txt"
            destination.write_text("keep this file", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                download_dataset(destination)

            self.assertEqual(destination.read_text(encoding="utf-8"), "keep this file")


if __name__ == "__main__":
    unittest.main()
