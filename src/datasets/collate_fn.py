import torch
def collate_fn(batch, tokenizer):
    # Текст для токенизации: ингредиенты + масса
    texts = [
        f"Ingredients: {item['ingredients']}. Mass: {item['mass']} g"
        for item in batch
    ]

    # Батч картинок
    images = torch.stack([item["image"] for item in batch])

    # Масса как отдельный числовой признак
    masses = torch.tensor([item["mass"] for item in batch], dtype=torch.float32)

    # Целевая переменная — калории
    calories = torch.tensor([item["calories"] for item in batch], dtype=torch.float32)

    # Токенизация текста
    tokenized_input = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )

    return {
        "image": images,
        "mass": masses,
        "calories": calories,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"]
    }