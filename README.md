# Generative AI with LSTM — Shakespeare Text Generation

## Project overview

This project trains a word-level next-token language model using TensorFlow/Keras and an LSTM. Given a sequence of words, the model estimates a probability distribution for the next word; generation repeatedly samples from that distribution to continue one or more prompts.

Objectives:

- Download and preprocess a reproducible public-domain text.
- Prepare aligned word-context and next-word targets.
- Train a regularized LSTM with a chronological validation split.
- Generate text from multiple prompts using temperature sampling.
- Save the vocabulary, best checkpoint, training metrics, plot, and generated samples.

## How the generator works

An input context of `sequence_length` word IDs passes through an embedding and an LSTM. The LSTM summarizes the context into a state; a dense softmax layer turns that state into one probability per vocabulary item. During generation, the chosen next word is appended to the context and the process repeats.

## Dataset and attribution

The dataset is *The Complete Works of William Shakespeare*, Project Gutenberg ebook 100. Source: [https://www.gutenberg.org/cache/epub/100/pg100.txt](https://www.gutenberg.org/cache/epub/100/pg100.txt). The source URL was verified during project setup. Shakespeare's works are public domain; Project Gutenberg distributes this ebook subject to its [license and terms](https://www.gutenberg.org/policy/license.html). Users should check the terms applicable in their jurisdiction.

Download the text into the default local path:

```powershell
python main.py download
```

This writes `data/shakespeare.txt`. Existing files are preserved; to intentionally replace the destination, use `python main.py download --overwrite`. Downloaded data is excluded from Git.

## Preprocessing

`src/preprocessing.py` removes content before and after standard Project Gutenberg start/end markers when found, lowercases text, replaces Unicode punctuation with spaces, and splits on whitespace. It does not remove stop words, normalize spelling, or stem words. A vocabulary capped at 10,000 entries by default is built using training text only; `<unk>` has ID 0 and represents rare, held-out, or unseen words.

Targets are next-token IDs. An example context `[to, be]` is paired with the immediately following word `or`. Samples are chronological. The split happens before vocabulary fitting and sequence construction: training targets end before the split and validation targets begin at the split; the initial validation context can include preceding training text, but no validation target is used for training. Sequence counts can be capped for a smaller first run.

## Model architecture

`src/model.py` builds this Keras model:

1. **Input** — fixed-length integer token sequence.
2. **Embedding** — learns a dense vector for each vocabulary ID instead of feeding sparse word IDs directly.
3. **LSTM** — learns sequential context and recency relationships.
4. **Dropout** — regularizes the LSTM output during training.
5. **Dense + softmax** — predicts a probability distribution over the full vocabulary for the next word.

The model uses Adam and sparse categorical crossentropy because each target is one integer token ID. Early stopping monitors validation loss and restores the best weights; model checkpointing saves the best validation-loss model.

## Install and run (Windows PowerShell)

Python 3.11 is used below; the manifest pins TensorFlow 2.15.x and compatible NumPy `<2`.

```powershell
Set-Location "C:\Users\Asus\OneDrive\Documents\Desktop\demo\Text-Generation"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py download
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that shell and activate again. A short end-to-end smoke training run (one epoch and a few hundred samples) can be run without overwriting the default training artifacts:

```powershell
python main.py train --experiment-name smoke --epochs 1 --patience 0 --max-train-sequences 256 --max-validation-sequences 64 --batch-size 32 --model models\smoke_model.keras --vocabulary models\smoke_vocabulary.json --history outputs\smoke_history.json --plot outputs\smoke_loss.png --metadata outputs\smoke_metadata.json
python main.py generate --model models\smoke_model.keras --vocabulary models\smoke_vocabulary.json --seeds "To be" "All the world's" --num-tokens 30 --temperature 0.8 --output outputs\smoke_samples.txt
```

Run the default training after installing requirements and downloading data:

```powershell
python main.py train
```

To use more data or longer training, adjust the flags shown by `python main.py train --help`. The defaults are sequence length 20, embedding dimension 64, 128 LSTM units, one LSTM layer, dropout 0.2, recurrent dropout 0, Adam learning rate 0.001, batch size 128, 5 epochs, early-stopping patience 2, up to 30,000 training and 5,000 validation sequences, vocabulary limit 10,000, validation fraction 0.1, and seed 42. `ReduceLROnPlateau` monitors validation loss, halves the learning rate after one non-improving epoch, and uses a minimum of `1e-5`.

Training prints final training and validation loss/accuracy and saves:

- `models/best_model.keras` — best checkpoint by validation loss.
- `models/vocabulary.json` — matching token-to-ID mapping.
- `models/best_model.manifest.json` — hashes and dimensions binding the checkpoint to its vocabulary.
- `outputs/training_history.json` — per-epoch loss/accuracy series.
- `outputs/loss_curve.png` — loss and accuracy plots generated from that run's history.
- `outputs/run_metadata.json` — experiment settings, actual best epoch/metrics, run duration, dataset hash, and model/vocabulary hashes.

Use unique paths for each experiment so histories and matching artifacts are not overwritten.

## Generate from multiple prompts

After training:

```powershell
python main.py generate --seeds "To be" "All the world's" "Once upon a time" --num-tokens 100 --temperature 0.8
```

This saves samples to `outputs/generated_samples.txt` and also prints them. The default generation context length (20) must match the training context length; specify `--sequence-length N` if training used a non-default value. A lower positive temperature concentrates probability on likely words; a higher value makes sampling more varied. Generation uses seed 42 by default; change it with `--seed`.

## Actual run results

### Bounded CPU training run (completed)

The following bounded run completed successfully on the local CPU with TensorFlow 2.15.1. It used the default seed (42), validation fraction (0.1), sequence length (20), embedding size (64), LSTM units (128), dropout (0.2), and vocabulary cap (10,000), with 4,096 training samples, 512 validation samples, batch size 64, a maximum of 2 epochs, and patience 1. The machine exposed no TensorFlow GPU devices. The training command took 12.28 seconds wall-clock including process startup, preprocessing, training, and plotting.

| Epoch | Training loss | Training accuracy | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 7.9866 | 0.0281 | 6.9638 | 0.0293 |
| 2 | 6.5495 | 0.0320 | 6.8711 | 0.0293 |

These are results from a deliberately small smoke-scale run, not a quality benchmark. Four prompts were then generated with 80 new tokens each, temperature 0.8, and sampling seed 42. The actual output is in `outputs/generated_samples.txt`. It demonstrates the current quality limitation: text is repetitive and often incoherent, with `<unk>` tokens and no restored punctuation.

The run can be reproduced after installation and dataset download with:

```powershell
python main.py train --epochs 2 --patience 1 --max-train-sequences 4096 --max-validation-sequences 512 --batch-size 64
python main.py generate --seeds "To be" "All the world's" "Once upon a time" "What light through yonder" --num-tokens 80 --temperature 0.8
```

The training history is stored in `outputs/training_history.json`; the plot at `outputs/loss_curve.png` was generated from that history. The saved checkpoint and vocabulary are present locally but Git-ignored. A longer default training run has not been performed.

## Validation-loss investigation

The original three-epoch 10,000-word experiment showed training loss decreasing (`7.0807 → 6.4652`) while validation loss rose (`6.6817 → 6.7775`). Validation accuracy rose (`0.0316 → 0.0414`), so that run showed worsening validation likelihood but not a decline in every validation metric. The earlier 64-unit/0.4-dropout trial had best validation loss 6.6861, slightly worse than the 128-unit/0.2-dropout baseline's 6.6817; it was not promoted.

### Dataset and vocabulary measurements

The downloaded source file is 5,834,878 bytes. After marker removal and cleaning, it contains 989,038 word tokens, split chronologically at token 890,134 (90% training, 10% validation). The training section has 23,515 unique word types. With a 10,000-entry cap (9,999 words plus `<unk>`), training-position coverage is 97.76%; 5.33% of validation token positions map to `<unk>`. A 5,000-entry cap (4,999 words plus `<unk>`) covers 94.37% of training positions and maps 9.23% of validation positions to `<unk>`. The frequency cap reduces output classes and softmax cost at the cost of treating more rare names/words as unknown.

### Controlled fixed-split runs (completed)

Both fresh experiments used the same dataset SHA-256, chronological split, seed 42, sequence length 20, embedding 64, 128 LSTM units, dropout 0.2, Adam learning rate 0.001, batch size 128, 30,000 training samples, 5,000 validation samples, maximum 5 epochs, early-stopping patience 2, and ReduceLROnPlateau (factor 0.5, patience 1). Only the vocabulary limit changed. Each stopped after three epochs; the epoch-1 checkpoint had the best validation loss.

| Experiment | Vocabulary entries | Best epoch | Best train loss / accuracy | Best validation loss / accuracy | Epoch-3 val loss / accuracy | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| A — control | 10,000 | 1 | 7.0807 / 3.10% | 6.6817 / 3.16% | 6.7580 / 3.86% | 50.33 s |
| B — reduced vocabulary | 5,000 | 1 | 6.6423 / 5.56% | 6.2509 / 9.48% | 6.2750 / 9.22% | 30.61 s |

The raw validation metrics are **not a like-for-like accuracy comparison** because B maps a larger share of target words to `<unk>` and predicts fewer output classes. As a diagnostic on the same sampled held-out targets where the target is one of the 4,999 shared known words (4,526 of 5,000 targets), each model's probabilities were restricted and renormalized over the shared vocabulary: conditional cross-entropy was 6.5272 for A and 6.5225 for B; shared-vocabulary top-1 accuracy was 3.49% for both. Thus B substantially improves its own capped-vocabulary task's raw metrics and CPU time, but shared-word discrimination is essentially unchanged in this single run. This is evidence to evaluate a cap around 5,000, not proof of universally better language modeling or generated text.

Three prompts were generated from each experiment's best checkpoint with 80 new tokens, temperature 0.8, and seed 42. Both outputs continue their prompts, but remain repetitive and weakly coherent; B has more `<unk>` tokens consistent with its increased OOV rate. Samples are in `outputs/accuracy_baseline_samples.txt` and `outputs/vocab_5000_samples.txt`.

### Experiment artifacts

- A: `outputs/accuracy_baseline_history.json`, `accuracy_baseline_plot.png`, `accuracy_baseline_metadata.json`, and `accuracy_baseline_samples.txt`.
- B: `outputs/vocab_5000_history.json`, `vocab_5000_plot.png`, `vocab_5000_metadata.json`, and `vocab_5000_samples.txt`.
- Matching models and vocabularies use corresponding `accuracy_baseline_*` and `vocab_5000_*` names in `models/` and are Git-ignored.
- Metadata records model and vocabulary SHA-256 hashes to identify the paired artifacts. Existing earlier histories and samples were not overwritten.

Reproduce A, B, and B's generation with:

```powershell
python main.py train --experiment-name accuracy_baseline --sequence-length 20 --validation-fraction 0.1 --embedding-dim 64 --lstm-units 128 --dropout 0.2 --learning-rate 0.001 --batch-size 128 --epochs 5 --patience 2 --max-train-sequences 30000 --max-validation-sequences 5000 --max-vocabulary-size 10000 --seed 42 --model models\accuracy_baseline_model.keras --vocabulary models\accuracy_baseline_vocabulary.json --history outputs\accuracy_baseline_history.json --plot outputs\accuracy_baseline_plot.png --metadata outputs\accuracy_baseline_metadata.json
python main.py train --experiment-name vocab_5000 --sequence-length 20 --validation-fraction 0.1 --embedding-dim 64 --lstm-units 128 --dropout 0.2 --learning-rate 0.001 --batch-size 128 --epochs 5 --patience 2 --max-train-sequences 30000 --max-validation-sequences 5000 --max-vocabulary-size 5000 --seed 42 --model models\vocab_5000_model.keras --vocabulary models\vocab_5000_vocabulary.json --history outputs\vocab_5000_history.json --plot outputs\vocab_5000_plot.png --metadata outputs\vocab_5000_metadata.json
python main.py generate --model models\vocab_5000_model.keras --vocabulary models\vocab_5000_vocabulary.json --seeds "To be" "All the world's" "Once upon a time" --num-tokens 80 --temperature 0.8 --seed 42 --output outputs\vocab_5000_samples.txt
```

## Reproducibility and limitations

- Set `--seed` to control Python/NumPy/TensorFlow and sampling randomness. Hardware kernels may still introduce small nondeterministic differences.
- This is a small word-level model, not a modern large language model. Limited context, vocabulary truncation, data scale, and one-token-at-a-time sampling can produce repetition, `<unk>` tokens, and ungrammatical text.
- Validation is chronological, but a book's later text can differ in style/content from earlier text, so validation estimates may vary with the split.
- Model inference requires the saved vocabulary and a `.keras` checkpoint with matching dimensions. New training runs save a hash manifest beside the model, and generation verifies the pair before loading the checkpoint. Older artifacts without a manifest produce a warning and can only be checked by dimensions. The program loads Keras models with `safe_mode=True`; only load files from trusted sources. Models and vocabulary are ignored by Git.
- TensorFlow CPU training may be slow; GPU availability and acceleration depend on local hardware, drivers, and TensorFlow platform support.

## Future improvements

Run a controlled comparison of sequence lengths or LSTM sizes with the same dataset split, seed, and sample limits; record validation loss and elapsed training time for both. This comparison has not been run. Other extensions include learning-rate scheduling, top-k/top-p sampling, better punctuation restoration, and a larger validated vocabulary.

## Git and artifact policy

Virtual environments, datasets, and model checkpoints are excluded from Git. The generated samples, actual training history, and loss plot are allowed as submission outputs; they should only be committed once produced and reviewed. The checkpoint produced by this run is 24,374,259 bytes (about 24.4 MB decimal), below GitHub's usual 100 MB per-file limit, but is intentionally not part of the source submission. Regenerate it with the documented training command, or distribute it separately through Git LFS or a trusted release/artifact store and publish a verified download URL and checksum. Do not commit secrets or untrusted serialized models.

## Tests

```powershell
python -m unittest discover -s tests -v
python -m compileall -q main.py src tests
```
