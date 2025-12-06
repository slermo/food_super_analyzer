import torch
import torch.nn as nn

def validate(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    criterion = nn.MSELoss()
    n_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device),
            }
            targets = batch['calories'].to(device)

            outputs = model(**inputs)
            outputs = outputs.squeeze()
            
            loss = criterion(outputs, targets)
            val_loss += loss.item() * targets.size(0)
            n_samples += targets.size(0)
    
    avg_val_loss = val_loss / n_samples
    return avg_val_loss