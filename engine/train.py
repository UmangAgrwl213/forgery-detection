# engine/train.py

import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None, autocast_dtype=torch.float16):
    model.train()

    total_loss = 0.0
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    pbar = tqdm(loader, desc="Training", leave=False)

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Forward with Mixed Precision
        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)

        # Backward with Mixed Precision
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Unscale before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        # Update OneCycleLR per batch
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
        })

    avg_loss = total_loss / len(loader)
    return avg_loss
