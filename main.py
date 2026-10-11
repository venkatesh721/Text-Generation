"""Command-line interface for downloading, training, and generating text."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from src.generation import generate_text
from src.preprocessing import (
    Vocabulary,
    download_dataset,
    load_tokens,
    prepare_train_validation,
)


ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser and its subcommands."""
    parser = argparse.ArgumentParser(description="Train an LSTM to generate Shakespeare-like text.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download Shakespeare from Project Gutenberg.")
    download_parser.add_argument("--output", type=Path, default=ROOT / "data" / "shakespeare.txt")
    download_parser.add_argument(
        "--overwrite", action="store_true", help="Replace the output file if it already exists."
    )

    train_parser = subparsers.add_parser("train", help="Prepare data and train the LSTM.")
    train_parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "shakespeare.txt")
    train_parser.add_argument("--sequence-length", type=int, default=20)
    train_parser.add_argument("--validation-fraction", type=float, default=0.1)
    train_parser.add_argument("--embedding-dim", type=int, default=64)
    train_parser.add_argument("--lstm-units", type=int, default=128)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--recurrent-dropout", type=float, default=0.0)
    train_parser.add_argument("--lstm-layers", type=int, choices=(1, 2), default=1)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--patience", type=int, default=2)
    train_parser.add_argument("--max-train-sequences", type=int, default=30000)
    train_parser.add_argument("--max-validation-sequences", type=int, default=5000)
    train_parser.add_argument("--max-vocabulary-size", type=int, default=10000)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--model", type=Path, default=ROOT / "models" / "best_model.keras")
    train_parser.add_argument("--vocabulary", type=Path, default=ROOT / "models" / "vocabulary.json")
    train_parser.add_argument("--history", type=Path, default=ROOT / "outputs" / "training_history.json")
    train_parser.add_argument("--plot", type=Path, default=ROOT / "outputs" / "loss_curve.png")
    train_parser.add_argument(
        "--metadata", type=Path, default=ROOT / "outputs" / "run_metadata.json"
    )
    train_parser.add_argument("--experiment-name", default="baseline")

    generate_parser = subparsers.add_parser("generate", help="Generate text from one or more seed phrases.")
    generate_parser.add_argument("--seeds", nargs="+", required=True)
    generate_parser.add_argument("--num-tokens", type=int, default=100)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--sequence-length", type=int, default=20)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--model", type=Path, default=ROOT / "models" / "best_model.keras")
    generate_parser.add_argument("--vocabulary", type=Path, default=ROOT / "models" / "vocabulary.json")
    generate_parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "generated_samples.txt"
    )
    return parser


def run_download(args: argparse.Namespace) -> None:
    destination = download_dataset(args.output, overwrite=args.overwrite)
    print(f"Downloaded Project Gutenberg Shakespeare text to {destination}")


def run_training(args: argparse.Namespace) -> None:
    started_at = time.perf_counter()
    tokens = load_tokens(args.dataset)
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for training. Install dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    (
        vocabulary,
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
    ) = prepare_train_validation(
        tokens,
        args.sequence_length,
        args.validation_fraction,
        args.max_train_sequences,
        args.max_validation_sequences,
        args.max_vocabulary_size,
    )
    from src.model import create_model, train_model

    model = create_model(
        vocabulary.size,
        args.sequence_length,
        args.embedding_dim,
        args.lstm_units,
        args.dropout,
        args.learning_rate,
        args.lstm_layers,
        args.recurrent_dropout,
    )
    history = train_model(
        model,
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
        vocabulary,
        args.model,
        args.vocabulary,
        args.history,
        args.plot,
        args.epochs,
        args.batch_size,
        args.patience,
        args.seed,
    )
    if not history.get("loss") or not history.get("val_loss"):
        raise RuntimeError("Training finished without loss metrics; check the dataset and configuration.")
    best_epoch = int(np.argmin(history["val_loss"]))
    metadata = {
        "experiment_name": args.experiment_name,
        "dataset_path": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "dataset_token_count": len(tokens),
        "training_token_positions": int(len(tokens) * (1 - args.validation_fraction)),
        "validation_token_positions": len(tokens)
        - int(len(tokens) * (1 - args.validation_fraction)),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "sequence_length": args.sequence_length,
        "training_samples": int(len(train_targets)),
        "validation_samples": int(len(validation_targets)),
        "vocabulary_size": int(vocabulary.size),
        "maximum_vocabulary_size": args.max_vocabulary_size,
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "embedding_dim": args.embedding_dim,
        "lstm_units": args.lstm_units,
        "lstm_layers": args.lstm_layers,
        "dropout": args.dropout,
        "recurrent_dropout": args.recurrent_dropout,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "epochs_completed": len(history["loss"]),
        "early_stopping_patience": args.patience,
        "early_stopping_monitor": "val_loss",
        "reduce_lr_on_plateau": True,
        "reduce_lr_factor": 0.5,
        "reduce_lr_patience": 1,
        "reduce_lr_minimum": 1e-5,
        "best_checkpoint_epoch": best_epoch + 1,
        "best_validation_loss": history["val_loss"][best_epoch],
        "validation_accuracy_at_best_checkpoint": history["val_accuracy"][best_epoch],
        "training_loss_at_best_checkpoint": history["loss"][best_epoch],
        "training_accuracy_at_best_checkpoint": history["accuracy"][best_epoch],
        "elapsed_seconds": time.perf_counter() - started_at,
        "model_path": str(args.model),
        "vocabulary_path": str(args.vocabulary),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "vocabulary_sha256": hashlib.sha256(
            args.vocabulary.read_bytes()
        ).hexdigest(),
        "history_path": str(args.history),
        "plot_path": str(args.plot),
    }
    manifest_path = args.model.with_suffix(".manifest.json")
    manifest = {
        "experiment_name": args.experiment_name,
        "model_sha256": metadata["model_sha256"],
        "vocabulary_sha256": metadata["vocabulary_sha256"],
        "sequence_length": args.sequence_length,
        "vocabulary_size": vocabulary.size,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metadata["model_manifest_path"] = str(manifest_path)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Training samples: {len(train_targets)}; validation samples: {len(validation_targets)}")
    print(
        f"Final training loss: {history['loss'][-1]:.4f}; "
        f"accuracy: {history['accuracy'][-1]:.4f}"
    )
    print(
        f"Final validation loss: {history['val_loss'][-1]:.4f}; "
        f"accuracy: {history['val_accuracy'][-1]:.4f}"
    )
    print(f"Best model: {args.model}")
    print(f"Training history: {args.history}")
    print(f"Loss curve: {args.plot}")
    print(f"Best checkpoint epoch: {best_epoch + 1}")
    print(f"Run metadata: {args.metadata}")


def run_generation(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {args.model}. Run 'python main.py train' first."
        )
    if args.model.suffix.lower() != ".keras":
        raise ValueError(
            "Only trusted Keras .keras model files are supported for safe loading."
        )
    if not args.vocabulary.is_file():
        raise FileNotFoundError(
            f"Vocabulary not found at {args.vocabulary}. Run 'python main.py train' first."
        )
    vocabulary = Vocabulary.load(args.vocabulary)
    manifest_path = args.model.with_suffix(".manifest.json")
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not read model/vocabulary manifest at {manifest_path}: {exc}"
            ) from exc
        expected_hashes = {
            "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "vocabulary_sha256": hashlib.sha256(
                args.vocabulary.read_bytes()
            ).hexdigest(),
        }
        for key, actual_hash in expected_hashes.items():
            if manifest.get(key) != actual_hash:
                raise ValueError(
                    f"Artifact mismatch: {key} in {manifest_path} does not match "
                    f"the selected model and vocabulary."
                )
        if manifest.get("vocabulary_size") != vocabulary.size:
            raise ValueError(
                f"Vocabulary size in {manifest_path} does not match "
                f"{args.vocabulary}."
            )
        if manifest.get("sequence_length") != args.sequence_length:
            raise ValueError(
                f"Sequence length in {manifest_path} does not match "
                f"--sequence-length {args.sequence_length}."
            )
    else:
        warnings.warn(
            f"No pairing manifest found at {manifest_path}; model/vocabulary "
            "compatibility can only be checked by dimensions. Retrain together "
            "to create a verifiable manifest.",
            RuntimeWarning,
            stacklevel=2,
        )

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for generation. Install dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    try:
        model = tf.keras.models.load_model(
            args.model,
            compile=False,
            safe_mode=True,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not safely load the Keras model at {args.model}: {exc}"
        ) from exc

    input_shape = model.input_shape
    output_shape = model.output_shape
    if (
        not isinstance(input_shape, tuple)
        or len(input_shape) != 2
        or input_shape[1] is None
    ):
        raise ValueError(f"Model has an unsupported input shape: {input_shape}.")
    if (
        not isinstance(output_shape, tuple)
        or len(output_shape) != 2
        or output_shape[-1] != vocabulary.size
    ):
        raise ValueError(
            "Model output dimension does not match vocabulary size "
            f"({output_shape} vs {vocabulary.size})."
        )
    if args.sequence_length != input_shape[1]:
        raise ValueError(
            f"Generation sequence length {args.sequence_length} does not match "
            f"the model input length {input_shape[1]}; pass "
            f"--sequence-length {input_shape[1]}."
        )
    rng = np.random.default_rng(args.seed)

    generated_samples = []
    for index, seed_text in enumerate(args.seeds, start=1):
        generated = generate_text(
            model,
            vocabulary,
            seed_text,
            args.sequence_length,
            args.num_tokens,
            args.temperature,
            rng,
        )
        generated_samples.append(f"Seed {index}: {seed_text}\n{generated}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(generated_samples) + "\n", encoding="utf-8")
    print(f"Generated {len(generated_samples)} sample(s) in {args.output}")
    for sample in generated_samples:
        print(f"\n{sample}\n")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "download":
            run_download(args)
        elif args.command == "train":
            run_training(args)
        else:
            run_generation(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
