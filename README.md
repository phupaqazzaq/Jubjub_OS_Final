# Thai Harassment Detection — OS-Optimized Pipeline

## Project Overview
A Thai text toxicity classifier that demonstrates OS optimization concepts:
CPU scheduling, memory management, multiprocessing, synchronization, and I/O management.

**Dataset:** Toxicity Multilingual Binary Classification Dataset (2025) — filtered for Thai  
**Model:** TF-IDF + Keyword Dictionary + Logistic Regression  
**Training Accuracy:** 89.5% | **Test Recall:** 50.4% on 1,889 harmful prompts

---

## Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/thai-harassment-os.git
cd thai-harassment-os

# Install all dependencies
pip install scikit-learn pythainlp pyarrow scipy pandas
```

If running on **Google Colab**, these are mostly pre-installed. Just run:
```bash
!pip install pythainlp
```

If running on **VS Code / local machine**:
```bash
pip install scikit-learn pythainlp pyarrow scipy pandas
```

---

## Command Reference

### Core Pipeline

| Command | What it does |
|---------|-------------|
| `python src/train_model.py` | Train the model from scratch. Benchmarks mmap vs read, picks the faster one. Tokenizes with multiprocessing. Saves model with fsync. Outputs accuracy and classification report. |
| `python src/predict.py` | Run demo predictions on 5 sample Thai texts. Shows toxic/non-toxic with confidence. |
| `python src/predict.py --interactive` | Interactive mode. Type any Thai text and get instant toxic/non-toxic prediction. Type `quit` to exit. |
| `python src/predict.py "ไอ้บ้า มึงไปตายซะ"` | Predict a single text from command line. Shows label, confidence, probabilities, and latency. |
| `python src/predict.py --file input.txt` | Predict all lines in a text file. Shows per-line results and summary. |
| `python src/evaluate.py` | Evaluate model on the test set (1,889 harmful prompts). Shows overall recall, recall by risk area, recall by harm type, and worst misses. Requires `data/test.parquet`. |

### OS Benchmarks

| Command | What it does |
|---------|-------------|
| `python src/pipeline.py` | Run the full OS-optimized pipeline: mmap/read benchmark → multiprocessing tokenization → atomic write. Shows timing for each step. |
| `python src/pipeline.py --all-benchmarks` | Run the pipeline + ALL individual OS benchmarks below in sequence. Use this to generate all demo screenshots at once. |
| `python src/data_loader.py` | **Memory Management** — Benchmark mmap() vs read() vs chunked read (512B, 64KB). Shows which is fastest for the 7MB dataset and why. |
| `python src/preprocessor.py --benchmark` | **Process Management** — Benchmark 1 vs 2 vs 4 vs 8 workers. Shows speedup, efficiency, and per-worker chunk sizes. Demonstrates fork(). |
| `python src/sync_queue.py` | **Synchronization** — Benchmark bounded buffer with different sizes (10, 50, 100, 500, 1000). Producer-consumer pattern with mutex + semaphores. Shows which buffer size is fastest. |
| `python src/io_benchmark.py` | **I/O Management** — Benchmark write strategies: buffered, unbuffered, fsync per row, fsync per 100, fsync per 1000. Shows the durability vs speed trade-off. |
| `python src/scheduler_demo.py` | **CPU Scheduling** — Benchmark CPU affinity (pinned vs unpinned) and context switch overhead (1 vs 4 vs 16 processes). Uses sched_setaffinity() and nice(). |
| `python src/file_manager.py` | **File Management** — Benchmark page cache (cold vs warm read), atomic vs direct write, and stat() syscall speed. |

### Performance Comparison

| Command | What it does |
|---------|-------------|
| `python src/benchmark_optimization.py` | **Before vs After** — Runs the entire pipeline twice: once naive (no OS optimization), once optimized. Compares timing stage-by-stage and shows total speedup. This is the teacher's recommended benchmark. |

---

## Project Structure

```
thai-harassment-os/
├── data/
│   ├── thai_toxicity.csv           # Training dataset (22,855 samples)
│   ├── toxic_keywords.csv          # 81 Thai profanity/slur keywords
│   ├── harm_intent_keywords.csv    # 70 harm-intent keywords (7 categories)
│   └── test.parquet                # Test set (1,889 harmful prompts)
├── model/                          # Created after training
│   ├── tfidf_model.pkl             # Trained Logistic Regression
│   ├── tfidf_vectorizer.pkl        # Fitted TF-IDF vocabulary
│   ├── toxic_keywords.json         # Keywords saved with model
│   ├── harm_intent_keywords.json   # Harm keywords saved with model
│   └── config.json                 # Model config and accuracy
├── src/
│   ├── train_model.py              # Model training
│   ├── predict.py                  # Prediction (single, interactive, file)
│   ├── evaluate.py                 # Test set evaluation
│   ├── pipeline.py                 # Full OS pipeline + benchmarks
│   ├── benchmark_optimization.py   # Before vs After comparison
│   ├── data_loader.py              # Memory management (mmap vs read)
│   ├── preprocessor.py             # Multiprocessing (fork)
│   ├── sync_queue.py               # Synchronization (bounded buffer)
│   ├── io_benchmark.py             # I/O management (fsync strategies)
│   ├── scheduler_demo.py           # CPU scheduling (affinity + nice)
│   └── file_manager.py             # File management (atomic write + cache)
└── README.md
```

---

## Quick Start (Colab)

```bash
!git clone https://github.com/YOUR_USERNAME/thai-harassment-os.git
%cd thai-harassment-os
!pip install pythainlp

# Train model
!python src/train_model.py

# Predict
!python src/predict.py --interactive

# Evaluate on test set
!python src/evaluate.py

# Before vs After benchmark
!python src/benchmark_optimization.py

# All OS benchmarks
!python src/pipeline.py --all-benchmarks
```

---

## OS Components → Rubric Mapping

| OS Component | File | System Calls | Rubric |
|---|---|---|---|
| Memory Management | `data_loader.py` | mmap, munmap, fstat, read | OS impl 30% |
| Process Management | `preprocessor.py` | fork, wait, getpid | OS impl 30% |
| Synchronization | `sync_queue.py` | Lock (futex), Semaphore (sem_wait/post) | OS impl 30% |
| CPU Scheduling | `scheduler_demo.py` | sched_setaffinity, nice | OS impl 30% |
| I/O Management | `io_benchmark.py` | open, write, fsync, close, unlink | Syscalls 20% |
| File Management | `file_manager.py` | stat, rename, mkstemp, unlink | Syscalls 20% |
| Performance Comparison | `benchmark_optimization.py` | All of the above | Trade-offs 20% |
