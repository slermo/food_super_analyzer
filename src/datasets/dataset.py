from torch.utils.data import Dataset
from PIL import Image
import timm
import albumentations as A
import numpy as np
import pandas as pd

# Класс-обёртка для трансформаций
class FoodDataset(Dataset):
    def __init__(self, 
                 transforms: A.Compose,
                 df_dish: pd.DataFrame,
                 df_ingr: pd.DataFrame):
        super().__init__()
        self.df_dish = df_dish
        self.df_ingr = df_ingr
        self.transforms = transforms

    def __len__(self):
        return len(self.df_dish)

    def __getitem__(self, idx):
        row = self.df_dish.iloc[idx]

        # Получение и трансформация фотографии
        image_path = f"data/images/{row['dish_id']}/rgb.png"
        img = Image.open(image_path).convert("RGB")
        img = self.transforms(image=np.array(img))["image"]
        
        dish_mass = row['total_mass']
        dish_ingr = get_ingredients_list(self.df_ingr, row)
        calories = row["total_calories"]
        return {
            "mass": dish_mass,
            "ingredients": dish_ingr,
            "image": img,
            "calories": calories
        }
    
def get_ingredients_list(df_ingr, dish_row):
    ingr_ids_raw = dish_row['ingredients'].split(';')
    # Преобразуем ingr_0000000X → X (int)
    ingr_ids = [int(x.replace("ingr_", "")) for x in ingr_ids_raw]
    # Ищем ингредиенты в df_ingr
    ing_rows = df_ingr[df_ingr['id'].isin(ingr_ids)]
    return ing_rows['ingr'].tolist()

def get_image_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], 
                                 cfg.input_size[2]), 
                                 p=1.0),
                A.RandomCrop(
                    height=cfg.input_size[1], 
                    width=cfg.input_size[2], 
                    p=1.0),
                A.Affine(
                    scale=(0.8, 1.2),
                    rotate=(-15, 15),
                    translate_percent=(-0.1, 0.1),
                    shear=(-10, 10),
                    fill=0,
                    p=0.8),
                A.CoarseDropout(
                    num_holes_range=(2, 8),
                    hole_height_range=(int(0.07 * cfg.input_size[1]),
                                       int(0.15 * cfg.input_size[1])),
                    hole_width_range=(int(0.1 * cfg.input_size[2]),
                                      int(0.15 * cfg.input_size[2])),
                                      fill=0,
                                      p=0.5),
                A.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2, 
                    saturation=0.2, 
                    hue=0.1, 
                    p=0.7),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=42,
        )
    else:
        transforms = A.Compose(
            [
                A.SmallestMaxSize(max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.CenterCrop(height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ]
        )

    return transforms