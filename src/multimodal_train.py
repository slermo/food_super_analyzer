import re
import torch
import torch.nn as nn
from conf import Config
from torch.utils.data import DataLoader
from src.multimodal_model import MultimodalModel
from torch.optim import AdamW
import src.multimodal_val as val

def train(config: Config,
          train_loader: DataLoader,
          val_loader: DataLoader,
          ):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Инициализация модели
    model = MultimodalModel(config).to(DEVICE)

    # Разморозка слоёв
    set_requires_grad(model.text_model, unfreeze_pattern=config.TEXT_MODEL_UNFREEZE)
    set_requires_grad(model.image_model, unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE)

    optimizer = AdamW([
        {'params': model.text_model.parameters(), 'lr': config.TEXT_LR},
        {'params': model.image_model.parameters(), 'lr': config.IMAGE_LR},
        {'params': model.regressor.parameters(), 'lr': config.REGRESSOR_LR}
    ])
    # Лосс функция MSE для задачи Регрессии
    criterion = nn.MSELoss()
    
    # Цикл обучения
    best_loss = -float('inf')
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            # Подготовка данных
            inputs = {
                'input_ids': batch['input_ids'].to(DEVICE),
                'attention_mask': batch['attention_mask'].to(DEVICE),
                'image': batch['image'].to(DEVICE),
                # 'mass': batch['mass'].to(DEVICE),
            }
            labels = batch['calories'].to(DEVICE)
            
            # Forward
            optimizer.zero_grad()
            logits = model(**inputs)
            loss = criterion(logits, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Валидация
        val_loss, val_mae = val.validate(model, val_loader, DEVICE)
        train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.EPOCHS} | avg_MSE: {train_loss:.2f} | Val MSE: {val_loss:.2f} | Val MAE: {val_mae:.2f}")
        
        # Сохранение лучшей модели
        if val_loss > best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), config.SAVE_PATH)

def set_requires_grad(module, 
                      unfreeze_pattern="", 
                      verbose=False):
    if len(unfreeze_pattern) == 0:
        for param, _ in module.named_parameters():
            param.requires_grad = False
        return

    pattern = re.compile(unfreeze_pattern)

    for name, param in module.named_parameters():
        if pattern.search(name):
            param.requires_grad = True
        if verbose:
            print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False