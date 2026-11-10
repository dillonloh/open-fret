import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

def aligned_truncate(midi_tokens, tab_tokens, max_tokens):
    """Cut a MIDI/tab pair at the same point in time so neither exceeds max_tokens.

    MIDI uses more tokens than tab for the same music (it also has note-offs and the
    CAPO/TUNING prefix), so truncating each to max_tokens separately leaves the tab
    target covering ~1.5x the time of the MIDI input. Both streams share the same
    TS_ (time shift) tokens in the same order, so cut MIDI at its last time-step
    boundary that fits and cut tab before the matching TS_.
    """
    if len(midi_tokens) <= max_tokens and len(tab_tokens) <= max_tokens:
        return midi_tokens, tab_tokens
    boundaries = [i for i in range(1, min(len(midi_tokens), max_tokens + 1))
                  if midi_tokens[i].startswith("TS_")]
    if len(midi_tokens) <= max_tokens:
        cut = len(midi_tokens)
    elif boundaries:
        cut = boundaries[-1]
    else:  # one time step longer than max_tokens: nothing to align on
        return midi_tokens[:max_tokens], tab_tokens[:max_tokens]
    steps = sum(1 for t in midi_tokens[:cut] if t.startswith("TS_"))
    tab_cut, seen = len(tab_tokens), 0
    for i, t in enumerate(tab_tokens):
        if t.startswith("TS_"):
            seen += 1
            if seen > steps:
                tab_cut = i
                break
    return midi_tokens[:cut], tab_tokens[:min(tab_cut, max_tokens)]

class TabDataset(Dataset):
    def __init__(self, file_list_path, data_dir, tokenizer, max_length=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(file_list_path, 'r', encoding='utf-8') as f:
            self.file_names = f.read().splitlines()
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_names[idx])
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Leave room for EOS
                midi_tokens, tab_tokens = aligned_truncate(data['midi'], data['tab'], self.max_length - 1)

                midi_text = " ".join(midi_tokens) + " " + self.tokenizer.eos_token
                tab_text = " ".join(tab_tokens) + " " + self.tokenizer.eos_token
                # Tokenize
                inputs = self.tokenizer(
                    midi_text, 
                    max_length=self.max_length, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt"
                )
                
                labels = self.tokenizer(
                    tab_text, 
                    max_length=self.max_length, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt"
                ).input_ids.squeeze()
                
                # Replace padding token id with -100 to ignore loss on padding
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                return {
                    'input_ids': inputs.input_ids.squeeze(),
                    'attention_mask': inputs.attention_mask.squeeze(),
                    'labels': labels
                }
        except Exception as e:
            # Return a dummy item if error (or handle better)
            return self.__getitem__((idx + 1) % len(self))

def get_dataloader(split, data_dir, tokenizer_path, batch_size=8, max_length=512):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    file_list = f"dataset/processed/{split}_files.txt"
    dataset = TabDataset(file_list, data_dir, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))

if __name__ == "__main__":
    # Test loading
    TOKENIZER_DIR = "dataset/processed/tab_tokenizer"
    DATA_DIR = "dataset/processed/individual"
    if os.path.exists(TOKENIZER_DIR):
        try:
            loader = get_dataloader("val", DATA_DIR, TOKENIZER_DIR, batch_size=2)
            batch = next(iter(loader))
            print("Batch keys:", batch.keys())
            print("Input shape:", batch['input_ids'].shape)
        except Exception as e:
            print(f"Error: {e}")
