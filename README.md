# GPT-2 

Educational repo which follows along from the [build-nanogpt](https://github.com/karpathy/build-nanogpt) repo and Andrej Karpathy's video [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU&feature=youtu.be).

# TODO

1. Fix DataLoader to work with Fineweb shards.
2. Validation logging.
3. Sample logging.
4. HellaSwag evals.
5. Save losses, evals, and samples to ndjson files.
6. Train.

# Path to training

### Tokenize the fineweb dataset.

`python fineweb.py`

### Download the HellaSwag evals.

`python hellaswag.py`

### Train the model.

Single gpu - `python train_gpt2.py`
Distributed Data Parallel training - `torchrun --standalone nproc_per_node=8 train_gpt2.py`