import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
from pathlib import Path
from tqdm.auto import tqdm
import argparse

# Import from our new modular structure
from model import UNet
from dataset import DehazeDataset
from utils import save_checkpoint, load_checkpoint, calculate_metrics, show_images

# --- Training and Validation Functions ---
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        hazy = batch['hazy'].to(device)
        clear = batch['clear'].to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast(device_type=device.type):
                dehazed = model(hazy)
                loss = criterion(dehazed, clear)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            dehazed = model(hazy)
            loss = criterion(dehazed, clear)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    return running_loss / len(dataloader)

def validate(model, dataloader, device, use_amp):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            hazy = batch['hazy'].to(device)
            clear = batch['clear'].to(device)
            
            if use_amp:
                with autocast(device_type=device.type):
                    dehazed = model(hazy)
            else:
                dehazed = model(hazy)
                
            for i in range(hazy.size(0)):
                psnr_val, ssim_val = calculate_metrics(clear[i], dehazed[i])
                total_psnr += psnr_val
                total_ssim += ssim_val
                
    avg_psnr = total_psnr / len(dataloader.dataset)
    avg_ssim = total_ssim / len(dataloader.dataset)
    
    normalized_psnr = avg_psnr / 40.0 
    combined_score = (normalized_psnr + avg_ssim) / 2.0
    
    return avg_psnr, avg_ssim, combined_score

# --- Main Training Script ---
def main(config):
    device = torch.device(config["DEVICE"])
    
    use_amp = device.type == 'cuda'
    if not use_amp:
        print("\n" + "="*50)
        print("WARNING: Training on CPU detected. This will be EXTREMELY slow.")
        print("For a practical training experience, a GPU is highly recommended.")
        print("="*50 + "\n")

    print(f"Using device: {device}")

    # --- Dataset and Output Paths ---
    data_root = Path(config["DATA_ROOT"])
    output_root = Path(config["OUTPUT_ROOT"])
    
    checkpoint_dir = output_root / "checkpoints"
    models_dir = output_root / "models"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Datasets and DataLoaders ---
    train_hazy_dir = data_root / config['TRAIN_SPLIT_NAME'] / "input"
    train_clear_dir = data_root / config['TRAIN_SPLIT_NAME'] / "target"
    test_hazy_dir = data_root / config['TEST_SPLIT_NAME'] / "input"
    test_clear_dir = data_root / config['TEST_SPLIT_NAME'] / "target"
    
    print(f"\nData Root: {data_root}")
    print(f"Training Hazy Dir: {train_hazy_dir}")
    print(f"Training Clear Dir: {train_clear_dir}")
    print(f"Testing Hazy Dir: {test_hazy_dir}")
    print(f"Testing Clear Dir: {test_clear_dir}")

    train_transform = transforms.Compose([
        transforms.Resize((config['IMAGE_SIZE'], config['IMAGE_SIZE'])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize((config['IMAGE_SIZE'], config['IMAGE_SIZE'])),
        transforms.ToTensor()
    ])
    
    train_dataset = DehazeDataset(hazy_dir=train_hazy_dir, clear_dir=train_clear_dir, transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, 
                              num_workers=config['NUM_WORKERS'], pin_memory=use_amp)

    test_dataset = DehazeDataset(hazy_dir=test_hazy_dir, clear_dir=test_clear_dir, transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False,
                             num_workers=config['NUM_WORKERS'], pin_memory=use_amp)
    
    # --- Model, Loss, Optimizer, Scheduler ---
    model = UNet(in_channels=3, out_channels=3).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    scaler = GradScaler(enabled=use_amp)

    # --- Checkpointing Logic ---
    checkpoint_path = checkpoint_dir / "checkpoint.pth"

    start_epoch = 0
    best_psnr = 0.0
    best_ssim = 0.0
    best_combined_score = 0.0
    
    if config["RESUME"] and checkpoint_path.exists():
        start_epoch, best_psnr, best_ssim, best_combined_score = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        start_epoch += 1
        print(f"Resuming training from epoch {start_epoch}")
        print(f"Previous best scores -> PSNR: {best_psnr:.4f}, SSIM: {best_ssim:.4f}, Combined: {best_combined_score:.4f}")
    else:
        print("Starting training from scratch.")

    print("\nStarting training...")
    try:
        for epoch in range(start_epoch, config['EPOCHS']):
            print(f"\n--- Epoch {epoch+1}/{config['EPOCHS']} ---")
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
            val_psnr, val_ssim, val_combined_score = validate(model, test_loader, device, use_amp)
            
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_combined_score)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < current_lr:
                print(f"  *** Learning rate reduced to {new_lr} ***")
            
            print(f"Epoch [{epoch+1}/{config['EPOCHS']}], Loss: {train_loss:.4f}, Val PSNR: {val_psnr:.4f}, Val SSIM: {val_ssim:.4f}, Combined Score: {val_combined_score:.4f}")
            
            # --- Save Best Models ---
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                print(f"  *** New best PSNR model: {best_psnr:.4f} ***")
                torch.save(model.state_dict(), models_dir / "best_model_psnr.pth")
                
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                print(f"  *** New best SSIM model: {best_ssim:.4f} ***")
                torch.save(model.state_dict(), models_dir / "best_model_ssim.pth")
                
            if val_combined_score > best_combined_score:
                best_combined_score = val_combined_score
                print(f"  *** New best Combined model: {val_combined_score:.4f} ***")
                torch.save(model.state_dict(), models_dir / "best_model_combined.pth")
            
            # --- Save Checkpoint ---
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_psnr': best_psnr,
                'best_ssim': best_ssim,
                'best_combined_score': best_combined_score
            }
            save_checkpoint(checkpoint, checkpoint_path)

            if (epoch + 1) % config['SHOW_IMAGES_EVERY_N_EPOCHS'] == 0:
                print("\nDisplaying sample results...")
                model.eval()
                with torch.no_grad():
                    sample_batch = next(iter(test_loader))
                    hazy_samples = sample_batch['hazy'].to(device)
                    clear_samples = sample_batch['clear'].to(device)
                    dehazed_samples = model(hazy_samples)
                    show_images(hazy_samples, clear_samples, dehazed_samples, num_images=config['NUM_IMAGES_TO_SHOW'])
                model.train()
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final checkpoint...")
        save_checkpoint(checkpoint, checkpoint_path)
    
    print("\nTraining finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Dehazing with U-Net')
    parser.add_argument('--data_root', type=str, default="./data", help='Root directory of dataset')
    parser.add_argument('--output_root', type=str, default="./output", help='Root directory to save outputs')
    parser.add_argument('--train_split', type=str, default="train", help='Training split name')
    parser.add_argument('--test_split', type=str, default="test", help='Testing split name')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256, help='Size of input images')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to use (cuda or cpu)')
    parser.add_argument('--show_every', type=int, default=10, help='Show images every N epochs')
    parser.add_argument('--num_show', type=int, default=3, help='Number of images to show')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    
    args = parser.parse_args()
    
    config = {
        "DATA_ROOT": args.data_root,
        "OUTPUT_ROOT": args.output_root,
        "TRAIN_SPLIT_NAME": args.train_split,
        "TEST_SPLIT_NAME": args.test_split,
        "EPOCHS": args.epochs,
        "BATCH_SIZE": args.batch_size,
        "LEARNING_RATE": args.lr,
        "IMAGE_SIZE": args.image_size,
        "DEVICE": args.device,
        "SHOW_IMAGES_EVERY_N_EPOCHS": args.show_every,
        "NUM_IMAGES_TO_SHOW": args.num_show,
        "RESUME": args.resume,
        "NUM_WORKERS": args.num_workers
    }
    
    main(config)