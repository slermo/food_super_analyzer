import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from conf import Config

def evaluate_model(model_class, 
                   pth_path: str, 
                   val_loader: DataLoader, 
                   config: Config):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Загрузка модели
    model = model_class(config).to(DEVICE)
    
    # Проверяем, что в файле лежит checkpoint
    checkpoint = torch.load(pth_path, map_location=DEVICE)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)


    eval_mae = 0.0
    all_items = []
    n_samples = 0

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(DEVICE),
                'attention_mask': batch['attention_mask'].to(DEVICE),
                'image': batch['image'].to(DEVICE),
                'mass': batch['mass'].to(DEVICE),
            }
            targets = batch['calories'].to(DEVICE)
            
            outputs = model(**inputs).squeeze()
            
            mae = torch.abs(outputs - targets)
            eval_mae += mae.sum().item()
            
            n_samples += targets.size(0)
            # Сохраняем сами элементы для визуализации
            for i in range(len(targets)):
                all_items.append({
                    'image': batch['image'][i],
                    'mass': batch['mass'][i].item() if 'mass' in batch else None,
                    'target': targets[i].item(),
                    'pred': outputs[i].item(),
                    'error': mae[i].item()
                })
    eval_mae /= n_samples
    print(f"EVAL MAE: {eval_mae:.2f}")
    # Сортируем по ошибке и выводим 5 худших предсказаний
    worst_items = sorted(all_items, key=lambda x: x['error'], reverse=True)[:5]
    
    # Топ 5 худших предскащаний
    plt.figure(figsize=(15,5))
    for i, item in enumerate(worst_items):
        img = item['image']
         # Убираем нормализацию
        if isinstance(img, torch.Tensor):
            img = img * 0.5 + 0.5
            img = img.permute(1, 2, 0).cpu().numpy()
        
        title = (
            f"Mass: {item['mass']} g\n"
            f"Target: {item['target']:.1f}, Pred: {item['pred']:.1f}\n"
            f"Error: {item['error']:.1f}\n"
        )
        
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(title, fontsize=9)
    
    plt.tight_layout()
    plt.show()