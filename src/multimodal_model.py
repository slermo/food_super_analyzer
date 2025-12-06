import timm
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from conf import Config

class MultimodalModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0 
        )
        # Приводим к одному к размеру
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)

        self.regressor = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM + 1, config.HIDDEN_DIM // 2), # +1 из-за добавления поля mass
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(config.HIDDEN_DIM // 2, 1)  # выход — калории
        )

    def forward(self, input_ids, attention_mask, image, mass):
        # текстовые эмбеддинги
        text_features = self.text_model(input_ids, attention_mask).last_hidden_state[:, 0, :]
        text_emb = self.text_proj(text_features)

        # Изображения
        image_features = self.image_model(image)
        image_emb = self.image_proj(image_features)

        # Умножаем фичи
        fused_emb = text_emb * image_emb
        
        fused_emb = torch.cat([fused_emb, mass.unsqueeze(1)], dim=1)
        # Предсказание калорий
        calories_pred = self.regressor(fused_emb)
        return calories_pred.squeeze(1) 
