# Thai Harassment Detection — OS-Optimized Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phupaqazzaq/Jubjub_OS_Final/blob/main/notebooks/Jubjub_OS_Final_Project.ipynb)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

A Thai text toxicity classifier built as the final project for **ITCS225 Principles of Operating Systems** (Faculty of ICT, Mahidol University) by team **JubJub**. The ML model is simple on purpose. The point of the project is the pipeline around it, which uses OS features to load, process and save data faster and more safely:

- **Memory management**: `mmap()` vs `read()`
- **Process management**: `fork()` worker pools
- **Synchronization**: bounded buffer with a mutex and semaphores
- **CPU scheduling**: CPU affinity and context-switch overhead
- **I/O management**: `fsync()` strategies
- **File management**: atomic writes, page cache

| | |
|---|---|
| **Dataset** | Toxicity Multilingual Binary Classification Dataset (2025), Thai subset: 22,855 labelled samples |
| **Model** | TF-IDF + Thai keyword dictionary + Logistic Regression (PyThaiNLP tokenizer) |
| **Validation accuracy** | **89.5%** (80/20 split: 18,282 train / 4,571 validation) |
| **Test recall** | **50.4%** (953 / 1,889 harmful prompts detected) |

📄 [Project report](docs/Project_Report.pdf) · 📊 [Presentation slides](docs/Final_Presentation_Slides.pdf) · 🎬 [Demo video](docs/Final_Project_Demo.mov)

---

## Two ways to run it

The project ships in two versions. Both run the same code in `src/`.

| Version | Where | Best for |
|---|---|---|
| **Google Colab** | [`notebooks/Jubjub_OS_Final_Project.ipynb`](notebooks/Jubjub_OS_Final_Project.ipynb) | Running everything in the browser with no setup |
| **Local machine** | [`src/`](src/) scripts | Running individual scripts and benchmarks from a terminal (VS Code, etc.) |

### Option 1 — Google Colab

1. Click the **Open in Colab** badge at the top of this page.
2. Choose **Runtime → Run all**.

The notebook clones this repository, installs the dependencies, then runs the pipeline, training, evaluation, prediction and all OS benchmarks in order.

### Option 2 — Local machine

Requires Python 3.10+.

```bash
git clone https://github.com/phupaqazzaq/Jubjub_OS_Final.git
cd Jubjub_OS_Final
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A trained model is already included in `model/`, so you can make predictions straight away:

```bash
python3 src/predict.py "ไอ้บ้า มึงไปตายซะ"
```

Or run the full workflow:

```bash
python3 src/pipeline.py              # OS-optimized data pipeline
python3 src/train_model.py           # train the model (~30 s on CPU)
python3 src/evaluate.py              # evaluate on the test set
python3 src/predict.py --interactive # type Thai text, get a prediction
```

> **Notes**
> - On macOS and Linux use `python3`. If you get `ModuleNotFoundError`, you are probably running a different Python than the one you installed the packages into.
> - CPU affinity (`sched_setaffinity`) only exists on Linux. On macOS and Windows, `scheduler_demo.py` still runs but skips pinning. Colab runs on Linux, so it shows the full result.
> - If loading the included model fails because of a scikit-learn version mismatch, retrain it with `python3 src/train_model.py`.

---

## Command reference

### Core pipeline

| Command | What it does |
|---|---|
| `python3 src/train_model.py` | Trains the model from scratch. Benchmarks `mmap` vs `read` and uses the faster one, tokenizes with multiprocessing, saves with `fsync`, and prints accuracy and a classification report. |
| `python3 src/predict.py` | Runs demo predictions on 5 sample Thai texts. |
| `python3 src/predict.py "<text>"` | Predicts one text: label, confidence, probabilities and latency. |
| `python3 src/predict.py --interactive` | Interactive mode. Type Thai text to get a prediction; type `quit` to exit. |
| `python3 src/predict.py --file input.txt` | Predicts every line of a text file and prints a summary. |
| `python3 src/evaluate.py` | Evaluates on the 1,889 harmful test prompts: overall recall, recall by risk area and harm type, and the worst misses. Saves `data/test_results.csv`. |

### OS benchmarks

| Command | OS topic | What it does |
|---|---|---|
| `python3 src/pipeline.py` | All | Full pipeline: `mmap`/`read` benchmark → multiprocessing tokenization → atomic write, with timing for each step. |
| `python3 src/pipeline.py --all-benchmarks` | All | The pipeline plus every benchmark below, in sequence. |
| `python3 src/data_loader.py` | Memory management | `mmap()` vs `read()` vs chunked reads (512 B, 64 KB). |
| `python3 src/preprocessor.py --benchmark` | Process management | 1, 2, 4 and up to 8 forked workers (capped at your CPU count): speedup, efficiency, chunk sizes. |
| `python3 src/sync_queue.py` | Synchronization | Producer–consumer bounded buffer (mutex + semaphores) with buffer sizes 10–1000. |
| `python3 src/io_benchmark.py` | I/O management | Buffered vs unbuffered writes, and `fsync` per row / per 100 / per 1000 rows. |
| `python3 src/scheduler_demo.py` | CPU scheduling | Pinned vs unpinned CPU affinity, and context-switch overhead with 1, 4 and 16 processes. |
| `python3 src/file_manager.py` | File management | Cold vs warm page-cache reads, atomic vs direct write, `stat()` speed. |
| `python3 src/benchmark_optimization.py` | Performance comparison | Runs the training pipeline naively and then OS-optimized, and compares each stage and the total speedup. |

### Optional: WangchanBERTa

`train_model.py --model bert` fine-tunes [WangchanBERTa](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased) instead of TF-IDF. It needs a GPU and `pip install torch transformers accelerate`. The fine-tuned model is saved to `model/wangchanberta/`, which is git-ignored because of its size.

---

## OS concepts → system calls

| OS component | File | System calls |
|---|---|---|
| Memory management | `data_loader.py` | `mmap`, `munmap`, `fstat`, `read` |
| Process management | `preprocessor.py` | `fork`, `wait`, `getpid` |
| Synchronization | `sync_queue.py` | Lock (`futex`), Semaphore (`sem_wait` / `sem_post`) |
| CPU scheduling | `scheduler_demo.py` | `sched_setaffinity`, `sched_getaffinity`, `getpid` |
| I/O management | `io_benchmark.py` | `open`, `write`, `fsync`, `close`, `unlink` |
| File management | `file_manager.py` | `stat`, `rename`, `mkstemp`, `unlink` |
| Performance comparison | `benchmark_optimization.py` | All of the above |

---

## Project structure

```
Jubjub_OS_Final/
├── notebooks/
│   └── Jubjub_OS_Final_Project.ipynb   # Google Colab version
├── src/                                # Local version: all scripts
│   ├── pipeline.py                     # Full OS pipeline + all benchmarks
│   ├── train_model.py                  # Model training (TF-IDF or BERT)
│   ├── predict.py                      # Prediction: single / interactive / file
│   ├── evaluate.py                     # Test set evaluation
│   ├── benchmark_optimization.py       # Before vs after comparison
│   ├── data_loader.py                  # Memory management (mmap vs read)
│   ├── preprocessor.py                 # Process management (fork)
│   ├── sync_queue.py                   # Synchronization (bounded buffer)
│   ├── scheduler_demo.py               # CPU scheduling (affinity + context switches)
│   ├── io_benchmark.py                 # I/O management (fsync strategies)
│   └── file_manager.py                 # File management (atomic write + cache)
├── data/
│   ├── thai_toxicity.csv               # Training set (22,855 samples)
│   ├── test.parquet                    # Test set (1,889 harmful prompts)
│   ├── toxic_keywords.csv              # 81 Thai profanity / slur keywords
│   └── harm_intent_keywords.csv        # 70 harm-intent keywords (7 categories)
├── model/                              # Trained model (rebuilt by train_model.py)
│   ├── tfidf_model.pkl                 # Logistic Regression
│   ├── tfidf_vectorizer.pkl            # Fitted TF-IDF vocabulary
│   ├── toxic_keywords.json
│   ├── harm_intent_keywords.json
│   └── config.json                     # Model settings and accuracy
├── docs/
│   ├── Project_Report.pdf
│   ├── Final_Presentation_Slides.pdf
│   └── Final_Project_Demo.mov
├── requirements.txt
└── README.md
```

The scripts also generate `data/thai_preprocessed.csv` (from `pipeline.py`) and `data/test_results.csv` (from `evaluate.py`). Both are git-ignored.

---

## Team JubJub

| Name | Nickname |
|---|---|
| Natdanai Chuwongapichat | Phupa |
| Uchukorn Ek-areejit | Korn |
| Duarn Kaeokertkhen | Du Du |
| Nathawat Burapolpiman | Q |
| Saharut Srithawong | Euro |

Presented to Aj. Kanoksak Wattanachote, ITCS225 Principles of Operating Systems, Faculty of Information and Communication Technology, Mahidol University.

---

## ⚠️ Content warning

The dataset and keyword lists contain offensive Thai language, including profanity, slurs and hate speech. They are included only to train and evaluate the classifier.
