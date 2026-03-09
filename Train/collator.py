import torch


def data_collator(features, processor=None):
    features = [f for f in features if f is not None]
    if not features:
        print("Empty batch after filtering None samples")
        return None

    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]
    labels = [f["labels"] for f in features]
    
    pixel_values = torch.cat([f["pixel_values"] for f in features], dim=0)
    image_grid_thw = torch.cat([f["image_grid_thw"] for f in features], dim=0) 

    pad_id = processor.tokenizer.pad_token_id if processor else 151643
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "labels": labels,
    }
