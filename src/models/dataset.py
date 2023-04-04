import torch
from torch.utils.data import Dataset



def collate_fn(batch, tokenizer, max_length):
    batch_x, batch_y = [list(x) for x in zip(*batch)]
    batch_x = tokenizer(
        batch_x,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    )
    batch_y = torch.tensor(batch_y)
    return batch_x, batch_y


class RelationExtractionDataset(Dataset):
    def __init__(self, dataset, model_config):
        self.dataset = dataset
        self.model_config = model_config

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        label_string = sample["label"]
        # TODO: fix this
        if label_string == "SUPPORTS":
            label_string = "ENTAILMENT"
        label = self.model_config.label2id[label_string]
        full_context = sample["full_context"]
        prompt_filled = sample["prompt_filled"]
        return (full_context, prompt_filled), label