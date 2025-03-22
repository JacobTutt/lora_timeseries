import math
import torch
from torch.utils.data import Dataset

# This is poo
# ------------------------ Define Dataset Class ------------------------ #
class TimeSeriesData(Dataset):
    """
    A PyTorch Dataset class that takes in a list of text sequences (e.g., time series in string form)
    and processes them using a tokeniser with optional chunking and padding.

    This is the class used on the preprocessed data which turns the numerical lists into strings with 
    commas and semicolons. This is then tokenised and chunked into the correct size for the model.

    Parameters
    ----------
    texts : List[str]
        The input data as a list of string sequences.
    tokenizer : PreTrainedTokenizer
        The tokenizer to be used (e.g., Qwen tokenizer).
    max_length : int, optional
        Maximum length of a token sequence. Defaults to 512.
    stride : int, optional
        The stride used for chunking the token sequences. Defaults to 256.
    """
    def __init__(self, texts, tokeniser, max_length=512, stride=256):
        # take in the tokensier that is created when loading the model, this will be used to convert from 
        # text to tokenised sequences which will then be broken into chunks
        self.tokenizer = tokeniser
        # The maximum length of the token sequences being produced
        self.max_length = max_length
        # The stride used to chunk the token sequences, typically longer for training
        self.stride = int(stride)
        # Run the internal method to process the sequences
        self.input_ids = self._process_sequences(texts)

    def _process_sequences(self, texts):
        all_input_ids = []

        # For each text sequence given
        for text in texts:
            # Tokenise the text sequence
            encoding = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            seq_ids = encoding.input_ids[0]

            for i in range(0, len(seq_ids), self.stride):
                # Chunk the token sequence into smaller pieces
                chunk = seq_ids[i : i + self.max_length]
                # Pad the chunk to the maximum length, this will be used on the last chunk 
                if len(chunk) < self.max_length:
                    chunk = torch.cat([chunk, torch.full((self.max_length - len(chunk),), self.tokenizer.pad_token_id)])

                # Add the chunk to the list of input IDs
                all_input_ids.append(chunk)

        return all_input_ids
    
    # The following two methods are required for the DataLoader to work
    def __len__(self):
        """
        Returns the total number of token sequences in the dataset.
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Returns the token sequence at the specified index.
        """
        return self.input_ids[idx]

# ------------------------ Example Use with DataLoader ------------------------ #

# Load the Qwen model and tokenizer
# train_dataset = TimeSeriesData(train_set, tokeniser, max_length=512, stride=256)
# val_dataset = TimeSeriesData(validation_set, tokeniser, max_length=512, stride=512)
# test_dataset = TimeSeriesData(test_set, tokeniser, max_length=512, stride=512)


# Convert the datasets into PyTorch DataLoader objects
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=4)
# test_loader = DataLoader(test_dataset, batch_size=4)