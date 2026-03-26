import torch
from tqdm import tqdm


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0

    for images, masks in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        val_loss += loss.item()

    return val_loss / len(loader)
