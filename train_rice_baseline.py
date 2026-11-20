import os
import sys
import time
import copy
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

sys.path.insert(0, 'src')
from config import cfg
from models.model import get_model
from datasets.reader import get_imglists
from datasets.GrainDataset import GrainDataset
from datasets.augment import get_transforms
from solver import get_optimizer, get_lr_scheduler, get_loss, get_current_lr, update_lr
from utils.misc import AverageMeter, accuracy

CLASS_NAMES = [
    "0_NOR (Normal)",
    "1_F&S (Fusarium/Shriveled)",
    "2_SD (Sprouted)",
    "3_MY (Moldy)",
    "4_AP (Attacked by Pests)",
    "5_BN (Broken)",
    "6_UN (Unripe)",
    "7_IM (Impurities)"
]

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    top2_meter = AverageMeter()

    for inputs, targets, _ in dataloader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        prec1, prec2 = accuracy(outputs.data, targets.data, topk=(1, 2))
        bs = inputs.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(prec1.item(), bs)
        top2_meter.update(prec2.item(), bs)

    return loss_meter.avg, acc_meter.avg, top2_meter.avg

def evaluate(model, dataloader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    top2_meter = AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            prec1, prec2 = accuracy(outputs.data, targets.data, topk=(1, 2))
            bs = inputs.size(0)
            loss_meter.update(loss.item(), bs)
            acc_meter.update(prec1.item(), bs)
            top2_meter.update(prec2.item(), bs)

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0) * 100

    return loss_meter.avg, acc_meter.avg, top2_meter.avg, macro_f1, weighted_f1, all_preds, all_targets

def export_to_onnx(model, save_path, device):
    try:
        model_to_export = model.module if hasattr(model, 'module') else model
        model_to_export.eval()
        dummy_input = torch.randn(1, 3, 224, 224, device=device)
        torch.onnx.export(
            model_to_export,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"--> Successfully exported ONNX model to: {save_path}")
    except Exception as e:
        print(f"Warning: Failed to export ONNX: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, default='configs/rice.yaml')
    parser.add_argument('--model-name', type=str, default='seresnet50')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1.2e-3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=12, help='Early stopping patience')
    parser.add_argument('--project-name', type=str, default='skripsi-rice-inspection')
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--no-wandb', action='store_true', default=False)
    args = parser.parse_args()

    cfg.update_from_file(args.config_file)
    cfg.MODEL.NAME = args.model_name
    cfg.TRAIN.EPOCHS = args.epochs
    cfg.TRAIN.BATCH = args.batch_size
    cfg.TEST.BATCH = args.batch_size
    cfg.DATASET.WORKERS = args.workers
    cfg.OPTIM.INIT_LR = args.lr

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = args.run_name or f"{args.model_name}_rice_full_{timestamp}"

    # Setup directories
    ckpt_dir = os.path.join(cfg.CHECKPOINTS, run_name)
    log_dir = os.path.join(cfg.LOG_PATH, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize wandb
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.project_name,
            name=run_name,
            config={
                "model": args.model_name,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "dataset": "Rice (31K)",
                "classes": 8,
                "loss": cfg.OPTIM.LOSS,
                "optimizer": cfg.OPTIM.NAME,
                "scheduler": cfg.OPTIM.LR_SCHEDULER,
            }
        )
        print(f"[W&B] Run initialized: {wandb.run.url}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Data loaders
    print("\nPreparing datasets...")
    train_imgs = get_imglists(root=cfg.DATASET.PATH, split="train", phase="train")
    val_imgs = get_imglists(root=cfg.DATASET.PATH, split="val", phase="train")
    test_imgs = get_imglists(root=cfg.DATASET.PATH, split="test", phase="train")

    train_tf, val_tf = get_transforms(RandBright_limit=0.2, RandBright_ratio=0.5, RandContra_limit=0.2, RandContra_ratio=0.5)
    train_dataset = GrainDataset(train_imgs, mode='train', transforms=train_tf)
    val_dataset = GrainDataset(val_imgs, mode='train', transforms=val_tf)
    test_dataset = GrainDataset(test_imgs, mode='test', transforms=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH, shuffle=True,
                              num_workers=cfg.DATASET.WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH, shuffle=False,
                            num_workers=cfg.DATASET.WORKERS, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH, shuffle=False,
                             num_workers=cfg.DATASET.WORKERS, pin_memory=True, drop_last=False)

    print(f"Train samples: {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"Val samples  : {len(val_dataset)} ({len(val_loader)} batches)")
    print(f"Test samples : {len(test_dataset)} ({len(test_loader)} batches)")

    # Model
    model = get_model(cfg, num_classes=cfg.DATASET.CLASS_NUMS)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, optim_name=cfg.OPTIM.NAME, learn_rate=cfg.OPTIM.INIT_LR)
    lr_scheduler = get_lr_scheduler(optimizer, lr_mode=cfg.OPTIM.LR_SCHEDULER)

    # Tracking
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    metrics_history = []
    csv_path = os.path.join(log_dir, "metrics.csv")

    print(f"\n================ START TRAINING: {run_name} ================")
    print(f"Total Epochs: {cfg.TRAIN.EPOCHS} | Batch Size: {cfg.TRAIN.BATCH} | LR: {cfg.OPTIM.INIT_LR}")

    start_total_time = time.time()

    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.EPOCHS):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # Train
        train_loss, train_acc, train_top2 = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_top2, val_macro_f1, val_weighted_f1, val_preds, val_targets = evaluate(
            model, val_loader, criterion, device
        )

        update_lr(lr_scheduler, cfg.OPTIM.LR_SCHEDULER, value=epoch)
        epoch_dur = time.time() - epoch_start

        # Record metrics
        row = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_top2": train_top2,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_top2": val_top2,
            "val_macro_f1": val_macro_f1,
            "val_weighted_f1": val_weighted_f1,
            "duration_sec": epoch_dur
        }
        metrics_history.append(row)
        pd.DataFrame(metrics_history).to_csv(csv_path, index=False)

        # W&B log
        if use_wandb:
            wandb.log(row)

        print(f"Epoch [{epoch+1:2d}/{cfg.TRAIN.EPOCHS}] ({epoch_dur:.1f}s) "
              f"Loss: {train_loss:.4f} -> {val_loss:.4f} | "
              f"Acc: {train_acc:.2f}% -> {val_acc:.2f}% | "
              f"Val Macro F1: {val_macro_f1:.2f}% (Weighted: {val_weighted_f1:.2f}%)")

        # Save latest model (for resume / safety)
        latest_path = os.path.join(ckpt_dir, "latest_model.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
            'val_acc': val_acc,
            'val_macro_f1': val_macro_f1,
            'config': cfg
        }, latest_path)

        # Check if best model (based on macro F1, robust for imbalanced classes)
        is_best = val_macro_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_macro_f1
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0

            best_ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'val_acc': best_val_acc,
                'val_macro_f1': best_val_f1,
                'val_weighted_f1': val_weighted_f1,
                'config': cfg
            }, best_ckpt_path)

            # Export ONNX for best model
            onnx_path = os.path.join(ckpt_dir, "best_model.onnx")
            export_to_onnx(model, onnx_path, device)
            print(f"  ★ New Best Val Macro F1: {best_val_f1:.2f}% (Acc: {best_val_acc:.2f}%) saved to {best_ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping] No improvement for {args.patience} epochs. Stopping at epoch {epoch+1}.")
                break

    total_time = time.time() - start_total_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation Epoch: {best_epoch} | Best Val Macro F1: {best_val_f1:.2f}% | Best Val Acc: {best_val_acc:.2f}%")

    # Final Test Set Evaluation
    print("\n================ EVALUATION ON TEST SET (3,100 IMAGES) ================")
    best_ckpt = torch.load(os.path.join(ckpt_dir, "best_model.pth"), map_location=device, weights_only=False)
    model_to_eval = get_model(cfg, num_classes=cfg.DATASET.CLASS_NUMS).to(device)
    model_to_eval.load_state_dict(best_ckpt['model_state_dict'])

    test_loss, test_acc, test_top2, test_macro_f1, test_weighted_f1, test_preds, test_targets = evaluate(
        model_to_eval, test_loader, criterion, device
    )

    print(f"Test Accuracy    : {test_acc:.2f}%")
    print(f"Test Macro F1    : {test_macro_f1:.2f}%")
    print(f"Test Weighted F1 : {test_weighted_f1:.2f}%")

    report = classification_report(test_targets, test_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)
    print("\nClassification Report on Test Set:")
    print(report)

    with open(os.path.join(log_dir, "test_report.txt"), "w") as f:
        f.write(f"Test Accuracy    : {test_acc:.2f}%\n")
        f.write(f"Test Macro F1    : {test_macro_f1:.2f}%\n")
        f.write(f"Test Weighted F1 : {test_weighted_f1:.2f}%\n\n")
        f.write(report)

    cm = confusion_matrix(test_targets, test_preds)
    np.save(os.path.join(log_dir, "confusion_matrix.npy"), cm)

    if use_wandb:
        wandb.summary["test_acc"] = test_acc
        wandb.summary["test_macro_f1"] = test_macro_f1
        wandb.summary["test_weighted_f1"] = test_weighted_f1
        wandb.summary["best_val_macro_f1"] = best_val_f1
        wandb.summary["best_val_acc"] = best_val_acc
        wandb.summary["best_epoch"] = best_epoch

        wandb.log({
            "conf_mat": wandb.plot.confusion_matrix(
                probs=None,
                y_true=test_targets,
                preds=test_preds,
                class_names=CLASS_NAMES
            )
        })
        wandb.finish()
        print("[W&B] Run finished and synced successfully.")

if __name__ == '__main__':
    main()
