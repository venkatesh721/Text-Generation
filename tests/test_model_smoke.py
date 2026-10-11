"""TensorFlow-backed model shape and one-batch smoke test."""

import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from main import run_generation
from src.preprocessing import Vocabulary

try:
    import tensorflow as tf
except ImportError:
    tf = None

from src.model import create_model, train_model


@unittest.skipIf(tf is None, "TensorFlow is not installed.")
class ModelSmokeTests(unittest.TestCase):
    def test_model_shape_matches_next_token_vocabulary(self) -> None:
        vocabulary_size = 7
        sequence_length = 4
        model = create_model(
            vocabulary_size=vocabulary_size,
            sequence_length=sequence_length,
            embedding_dim=8,
            lstm_units=8,
            dropout=0.1,
        )

        self.assertEqual(model.input_shape, (None, sequence_length))
        self.assertEqual(model.output_shape, (None, vocabulary_size))
        self.assertEqual(model.loss, "sparse_categorical_crossentropy")

        inputs = np.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.int32)
        targets = np.array([5, 6], dtype=np.int32)
        probabilities = model.predict(inputs, verbose=0)
        self.assertEqual(probabilities.shape, (2, vocabulary_size))
        np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0], atol=1e-5)
        self.assertTrue(np.all(probabilities >= 0))
        loss = model.train_on_batch(inputs, targets)
        self.assertTrue(np.isfinite(loss).all())

    def test_two_layer_model_preserves_vocab_sized_output(self) -> None:
        model = create_model(
            vocabulary_size=9,
            sequence_length=5,
            embedding_dim=8,
            lstm_units=6,
            dropout=0.2,
            lstm_layers=2,
        )

        self.assertEqual(model.input_shape, (None, 5))
        self.assertEqual(model.output_shape, (None, 9))
        self.assertEqual(sum(isinstance(layer, tf.keras.layers.LSTM) for layer in model.layers), 2)

    def test_training_uses_minimum_validation_loss_callbacks(self) -> None:
        vocabulary = Vocabulary.from_tokens(["a", "b"])
        model = create_model(
            vocabulary_size=vocabulary.size,
            sequence_length=1,
            embedding_dim=2,
            lstm_units=2,
            dropout=0.0,
        )
        history = {
            "loss": [1.0],
            "accuracy": [0.5],
            "val_loss": [1.2],
            "val_accuracy": [0.4],
        }

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with patch.object(
                model, "fit", return_value=SimpleNamespace(history=history)
            ) as fit:
                train_model(
                    model=model,
                    train_inputs=np.array([[1]], dtype=np.int32),
                    train_targets=np.array([1], dtype=np.int32),
                    validation_inputs=np.array([[1]], dtype=np.int32),
                    validation_targets=np.array([1], dtype=np.int32),
                    vocabulary=vocabulary,
                    model_path=root / "best.keras",
                    vocabulary_path=root / "vocabulary.json",
                    history_path=root / "history.json",
                    plot_path=root / "loss.png",
                    epochs=3,
                    batch_size=1,
                    patience=2,
                    seed=42,
                )

            callbacks = fit.call_args.kwargs["callbacks"]
            early_stopping = next(
                callback
                for callback in callbacks
                if isinstance(callback, tf.keras.callbacks.EarlyStopping)
            )
            checkpoint = next(
                callback
                for callback in callbacks
                if isinstance(callback, tf.keras.callbacks.ModelCheckpoint)
            )
            self.assertEqual(early_stopping.monitor, "val_loss")
            self.assertTrue(early_stopping.monitor_op(1.0, 2.0))
            self.assertFalse(early_stopping.monitor_op(2.0, 1.0))
            self.assertTrue(early_stopping.restore_best_weights)
            self.assertEqual(checkpoint.monitor, "val_loss")
            self.assertTrue(checkpoint.monitor_op(1.0, 2.0))
            self.assertFalse(checkpoint.monitor_op(2.0, 1.0))
            self.assertTrue(checkpoint.save_best_only)
            reduce_lr = next(
                callback
                for callback in callbacks
                if isinstance(callback, tf.keras.callbacks.ReduceLROnPlateau)
            )
            self.assertEqual(reduce_lr.monitor, "val_loss")
            self.assertEqual(reduce_lr.mode, "min")

    def test_training_saves_best_checkpoint_and_metrics(self) -> None:
        sequence_length = 3
        vocabulary = Vocabulary.from_tokens(["a", "b", "c", "d"])
        model = create_model(
            vocabulary_size=vocabulary.size,
            sequence_length=sequence_length,
            embedding_dim=4,
            lstm_units=4,
            dropout=0.0,
        )
        train_inputs = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.int32)
        train_targets = np.array([4, 1], dtype=np.int32)
        validation_inputs = np.array([[3, 4, 1]], dtype=np.int32)
        validation_targets = np.array([2], dtype=np.int32)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_path = root / "best.keras"
            history_path = root / "history.json"
            train_history = train_model(
                model=model,
                train_inputs=train_inputs,
                train_targets=train_targets,
                validation_inputs=validation_inputs,
                validation_targets=validation_targets,
                vocabulary=vocabulary,
                model_path=model_path,
                vocabulary_path=root / "vocabulary.json",
                history_path=history_path,
                plot_path=root / "loss.png",
                epochs=1,
                batch_size=2,
                patience=0,
                seed=42,
            )

            self.assertTrue(model_path.is_file())
            self.assertTrue(history_path.is_file())
            self.assertTrue((root / "loss.png").is_file())
            self.assertTrue((root / "vocabulary.json").is_file())
            self.assertIn("val_loss", train_history)


class ArtifactManifestTests(unittest.TestCase):
    def _run_generation(
        self,
        root: Path,
        sequence_length: int = 2,
        corrupt_model: bool = False,
    ) -> None:
        model_path = root / "model.keras"
        vocabulary_path = root / "vocabulary.json"
        output_path = root / "samples.txt"
        model_path.write_bytes(b"test model artifact")
        vocabulary = Vocabulary.from_tokens(["a", "b"])
        vocabulary.save(vocabulary_path)
        manifest_path = model_path.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(
                {
                    "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    "vocabulary_sha256": hashlib.sha256(
                        vocabulary_path.read_bytes()
                    ).hexdigest(),
                    "sequence_length": 2,
                    "vocabulary_size": vocabulary.size,
                }
            ),
            encoding="utf-8",
        )
        if corrupt_model:
            model_path.write_bytes(b"replaced model artifact")

        class GenerationModel:
            input_shape = (None, 2)
            output_shape = (None, vocabulary.size)

            def predict(self, inputs: np.ndarray, verbose: int = 0) -> np.ndarray:
                probabilities = np.zeros((1, vocabulary.size))
                probabilities[0, vocabulary.word_to_id["a"]] = 1.0
                return probabilities

        fake_tensorflow = SimpleNamespace(
            keras=SimpleNamespace(
                models=SimpleNamespace(load_model=lambda *args, **kwargs: GenerationModel())
            )
        )
        args = SimpleNamespace(
            model=model_path,
            vocabulary=vocabulary_path,
            sequence_length=sequence_length,
            seeds=["a"],
            seed=42,
            num_tokens=1,
            temperature=1.0,
            output=output_path,
        )
        with patch.dict("sys.modules", {"tensorflow": fake_tensorflow}):
            run_generation(args)

    def test_generation_accepts_matching_model_vocabulary_manifest(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._run_generation(root)
            self.assertIn("a a", (root / "samples.txt").read_text(encoding="utf-8"))

    def test_generation_rejects_manifest_sequence_length_mismatch(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "Sequence length"):
                self._run_generation(Path(temporary_directory), sequence_length=3)

    def test_generation_rejects_model_hash_mismatch(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "model_sha256"):
                self._run_generation(
                    Path(temporary_directory),
                    corrupt_model=True,
                )


if __name__ == "__main__":
    unittest.main()
