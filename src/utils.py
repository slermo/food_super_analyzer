import matplotlib.pyplot as plt
import pandas as pd
from src.datasets.dataset import FoodDataset
from conf import Config
import timm
import torch

def plot_images(
    dataset: FoodDataset,
    start_ind: int = 0,
    end_ind: int = 5,):

    n_images = max(1,end_ind - start_ind)

    plt.figure(figsize=(2 * n_images, 6))
    for i, idx in enumerate(range(start_ind, end_ind)):
        item = dataset[idx]

        img = item["image"]
        mass = item["mass"]
        ingredients_text = item["ingredients"]
        calories = item["calories"]

        # Убираем нормализацию
        if isinstance(img, torch.Tensor):
            img = img * 0.5 + 0.5
            img = img.permute(1, 2, 0).cpu().numpy()

        title = (
            f"Mass: {mass} g\n"
            f"Calories: {calories:.1f}\n"
            f"Rnd ingredient:\n{ingredients_text[0]}"
        )

        plt.subplot(1, n_images, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.show()