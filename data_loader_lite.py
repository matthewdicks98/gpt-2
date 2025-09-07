import os
import torch
import numpy as np

class DataLoaderLite:

    def __init__(
            self, 
            batch_size: int, 
            seq_len: int, 
            process_num: int = 0, 
            num_processes: int = 1,
            split: str = "train"
        ):
        """
        Takes in the batch size and sequence length and loads the data.

        :param batch_size: Batch size.
        :param seq_len: Sequence length.
        :param num_processes: Number of processes running.
        :param process_num: Current process number.
        :param split: Train or validation split.
        """
        
        # Set the params.
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_batch_tokens = batch_size * seq_len
        self.process_num = process_num
        self.num_processes = num_processes

        assert split in {"train", "val"}
        self.split = split

        # Set the directory where the token shards exist.
        project_dir = os.environ.get("PROJECT_DIR", os.path.expanduser("~"))
        self.data_dir = os.path.join(project_dir, ".fineweb")

        # Get the split and shards this process is responsible for.
        self.shards = sorted(os.listdir(self.data_dir))
        self.shards = [s for s in self.shards if split in s]

        # Set the current token buffer.
        self.tokens = None
        self.current_shard = 0
        self.current_pos = 0

        # Set everything to the beginning.
        self.reset()

        # Print the total tokens and batches per epoch.
        if process_num == 0:
            print(f"Total tokens: {len(self.shards) * int(1e8)}")
            print(f"Batches per epoch: {len(self.shards) * int(1e8) // (batch_size * seq_len * num_processes)}")

    def reset(self):
        """
        Reset the dataloader back to the beginning.
        """
        self.current_shard = 0
        self.current_pos = self.process_num * self.n_batch_tokens
        self.tokens = self.load_tokens(shard=self.shards[self.current_shard])

    def next_batch(self):
        """
        Gets the next batch of data.

        TODO: I think there may be issues with this dataloader.
                1. Checking if batch overflow might be wrong.
                2. Might leave tokens at the end of the batch on the table.

        :return: The next batch of data. Shape=(batch_size, seq_len).
        """

        # Shapes.
        batch_size, seq_len, n_batch_tokens = self.batch_size, self.seq_len, self.n_batch_tokens 

        # Get next batch.
        buf = self.tokens[self.current_pos:self.current_pos + n_batch_tokens + 1]
        x = buf[:-1].view(batch_size, seq_len)
        y = buf[1:].view(batch_size, seq_len)

        # Compute new pos.
        self.current_pos += self.n_batch_tokens * self.num_processes

        # Check if there are enough tokens to fit.
        if self.current_pos + self.n_batch_tokens + 1 > len(self.tokens):

            # Get the current shard with wrap around and load.
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(shard=self.shards[self.current_shard])

            # Init back to the start.
            self.current_pos = self.process_num * n_batch_tokens

        return x, y
    
    def load_tokens(self, shard: int):
        """
        Load the tokens from the given shard.

        :param shard: Index of the shard we want to load.
        :return: The tokens for the shard.
        """
        tokens = np.load(os.path.join(self.data_dir, f"shard_{self.split}_{shard}.npy"))
        tokens = torch.from_numpy(tokens).to(dtype=torch.long)
        return tokens