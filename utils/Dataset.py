import torch
from torch.utils.data import Dataset
##REAL DATASET- NOT USED
class ASRDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        audio = self.dataset[idx]['audio']['array']
        #print(self.dataset[idx])
        input_values = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).input_features
        #print(input_values)
        print(input_values.shape)
        return input_values.squeeze(0)

##USED - DUMMY DATASET WITH SAME LENGTH
class DummyDataset(Dataset):
    def __init__(self, num_samples, seq_length, num_channels=0):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_channels = num_channels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.num_channels > 0:
            return torch.randn(self.num_channels, self.seq_length)
        else:
            return torch.randn(self.seq_length)

# Define a dummy image  dataset
class DummyImageDataset(Dataset):  
    #init with num_samples, shape=(channels, height, width)
    def __init__(self, num_samples, image_shape):
        self.num_samples = num_samples
        self.shape = image_shape
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        #input, labels
        return torch.randint(0, 256, self.shape, dtype=torch.float32), torch.tensor(1, dtype=torch.long)
 
# Define a dummy dataset
class DummyBertDataset(Dataset):
    def __init__(self, num_samples, seq_length, vocab_size):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(101, 102, (self.seq_length,), dtype=torch.long),  # Integer inputs within vocab size
            'attention_mask': torch.ones(self.seq_length, dtype=torch.long),                       # Integer mask
            'token_type_ids': torch.zeros(self.seq_length, dtype=torch.long),                        # Integer mask
            'labels': torch.tensor(1, dtype=torch.long)                                            # Integer labels
        }
