import numpy as np
import multiprocessing as mp
import tiktoken
from datasets import load_dataset
import os
from tqdm import tqdm


# Init the tokenizer.
ENCODER = tiktoken.get_encoding("gpt2")
EOT = ENCODER._special_tokens["<|endoftext|>"]


def tokenize_fineweb(sample: str):
    """
    Download the fineweb-edu dataset and tokenize.

    ETA on this should be around 30 min for sample-10BT.

    :param sample: what sample from fineweb we want to download and tokenize.
    """
    
    # Download the Fineweb-edu dataset.
    fineweb = load_dataset("HuggingFaceFW/fineweb-edu", name=sample, split="train")

    # Create the directory to save the data to.
    project_dir = os.environ.get("PROJECT_DIR", os.path.expanduser("~"))
    data_dir = os.path.join(project_dir, ".fineweb")
    os.makedirs(data_dir, exist_ok=True)

    total_tokens = 0

    # Tokenize the fineweb dataset into shards of uint16s.
    max_shard_size = int(1e8)
    n_processes = max(1, os.cpu_count() // 2)
    with mp.Pool(processes=n_processes, maxtasksperchild=1_000_000) as pool:

        progress_bar = None

        # Create the token shards.
        shard_index = 0
        all_tokens_np = np.empty((max_shard_size,), dtype=np.uint16)
        token_count = 0        

        for tokens in pool.imap_unordered(tokenize_doc, fineweb, chunksize=16):

            if token_count + len(tokens) < max_shard_size:

                # Simply append tokens to current shard.
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)

                if progress_bar is None:
                    progress_bar = tqdm(
                        total=max_shard_size, 
                        unit="tokens", 
                        desc=f"Shard {shard_index}"
                    )
                progress_bar.update(len(tokens))
            
            else:

                # Set the split.
                split = "train" if shard_index != 0 else "val"

                remainder = max_shard_size - token_count
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]

                # Save the previous shards tokens.
                np.save(
                    file=os.path.join(data_dir, f"shard_{split}_{shard_index}.npy"),
                    arr=all_tokens_np
                )

                # Clear the buffers and set a new shard.
                shard_index += 1
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

                # Empty the progress bar.
                progress_bar.update(remainder)
                progress_bar = None

                total_tokens += all_tokens_np.shape[0]

        # Save the overflow.
        if token_count != 0:

            split = "val" if shard_index == 0 else "train"
            np.save(
                file=os.path.join(data_dir, f"shard_{shard_index}.npy"),
                arr=all_tokens_np[:token_count]
            )

            total_tokens += all_tokens_np.shape[0]

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
    tokenize_fineweb(sample="sample-10BT")
