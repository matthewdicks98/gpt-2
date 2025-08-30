import numpy as np
import multiprocessing as mp
import tiktoken
from datasets import load_dataset
import os
from tqdm import tqdm


# Init the tokenizer.
ENCODER = tiktoken.get_encoding("gpt2")
EOT = ENCODER._special_tokens["<|endoftext|>"]


def tokenize_fineweb():
    """
    Download the fineweb-edu dataset and tokenize.

    ETA on this should be around 30 min for sample-10BT.
    """
    
    # Download the Fineweb-edu dataset.
    fineweb = load_dataset("HuggingFaceFW/fineweb-edu", name="CC-MAIN-2013-20", split="train")

    # Create the directory to save the data to.
    data_dir = os.path.join(os.path.expanduser("~"), ".fineweb")
    os.makedirs(data_dir, exist_ok=True)

    total_tokens = 0

    # Tokenize the fineweb dataset into shards of uint16s.
    max_shard_size = int(1e8)
    n_processes = max(1, os.cpu_count() // 2)
    with (
        mp.Pool(processes=n_processes) as pool,
        tqdm(total=max_shard_size, ncols=80) as pbar,
    ):

        # Create the token shards.
        shard_index = 0
        n_tokens = 0
        shard_tokens = []
        for tokens in pool.imap(tokenize_doc, fineweb, chunksize=16):
            
            n_tokens += tokens.shape[0]
            total_tokens += tokens.shape[0]
            if n_tokens < max_shard_size:

                # Add to the current shard tokens.
                shard_tokens.append(tokens)

                # Update the progress bar.
                pbar.update(shard_tokens_np.shape[0])
                pbar.set_description(f"shard_{shard_index}")

            else:

                # Set the split.
                split = "train" if shard_index != 0 else "val"

                # Save the previous shards tokens.
                shard_tokens_np = np.vstack(shard_tokens)
                np.save(
                    file=os.path.join(data_dir, f"shard_{split}_{shard_index}.npy"),
                    arr=shard_tokens_np
                )

                # Clear the buffers and set a new shard.
                shard_index += 1
                shard_tokens = []
                n_tokens = 0

                # Empty the progress bar.
                pbar.n = 0
                pbar.set_description(f"shard_{shard_index}")

        # Save the overflow.
        if len(shard_tokens) > 0:
            shard_tokens_np = np.vstack(shard_tokens)
            np.save(
                file=os.path.join(data_dir, f"shard_{shard_index}.npy"),
                arr=shard_tokens_np
            )

    print(f"Total tokens - {total_tokens}")


def tokenize_doc(doc: dict):
    """
    Takes in a row of the fineweb dataset and tokenize the text.
    """
    # Tokenize the text.
    tokens = [EOT]
    text_tokens = ENCODER.encode_ordinary(text=doc["text"])
    tokens.extend(text_tokens)
    tokens_np = np.array(tokens)

    # Convert to uint16.
    assert (tokens_np >= 0).all() and (tokens_np < 2**16).all(), "Tokens unit16 Overflow."
    return tokens_np.astype(np.uint16)


if __name__ == "__main__":
    tokenize_fineweb()
