import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import albumentations as A
from albumentations import Resize, Normalize
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, 'src')
from config import cfg
from datasets.GrainDataset import GrainDataset
from datasets.reader import get_imglists
from models.model import get_model
from models.backbones.resnet import seresnet50

MEANS = (0.308562, 0.251994, 0.187898)[::-1]
STDS  = (0.240441, 0.197289, 0.149387)[::-1]

val_transform = A.Compose([
    Resize(224, 224),
    Normalize(mean=MEANS, std=STDS),
    ToTensorV2()
])

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

def load_checkpoint_model(model_path, device):
    raw = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and 'model_state_dict' in raw:
        state_dict = raw['model_state_dict']
        print(f"Loaded checkpoint dict (Epoch {raw.get('epoch')}, Val Macro F1: {raw.get('val_macro_f1'):.2f}%, Val Acc: {raw.get('val_acc'):.2f}%)")
        model = seresnet50(num_classes=8)
        model.load_state_dict(state_dict)
    elif hasattr(raw, 'module'):
        model = raw.module
    else:
        model = raw
    model = model.to(device)
    model.eval()
    return model

def evaluate(model_path, data_path, split="test", batch_size=64):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n=======================================================")
    print(f"Loading checkpoint: {model_path}")
    print(f"Evaluating split: {split} from: {data_path}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"=======================================================")

    model = load_checkpoint_model(model_path, device)

    test_imgs = get_imglists(data_path, split=split)
    test_dataset = GrainDataset(test_imgs, mode='test', transforms=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    print(f"Total test samples: {len(test_dataset)}")

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = np.mean(all_preds == all_targets) * 100
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0) * 100

    print(f"\n=================== EVALUATION RESULTS ===================")
    print(f"Overall Accuracy   : {acc:.2f}%")
    print(f"Macro F1-Score     : {macro_f1:.2f}%")
    print(f"Weighted F1-Score  : {weighted_f1:.2f}%")
    print(f"==========================================================\n")

    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)
    print("Classification Report on Test Set (3,100 Rice Images):")
    print(report)

    cm = confusion_matrix(all_targets, all_preds, labels=range(8))
    print("\nConfusion Matrix (Rows = Ground Truth, Columns = Predicted):")
    print(cm)

    # Save results to file
    out_dir = os.path.dirname(model_path)
    report_file = os.path.join(out_dir, f"test_evaluation_report_{split}.txt")
    with open(report_file, "w") as f:
        f.write(f"Model Checkpoint: {model_path}\n")
        f.write(f"Dataset Path    : {data_path}\n")
        f.write(f"Split           : {split}\n\n")
        f.write(f"Overall Accuracy   : {acc:.2f}%\n")
        f.write(f"Macro F1-Score     : {macro_f1:.2f}%\n")
        f.write(f"Weighted F1-Score  : {weighted_f1:.2f}%\n\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
    print(f"\nReport saved to: {report_file}")

    return acc, macro_f1, weighted_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='/home/master_core_ti/skripsi-grain/GrainSet/runs/checkpoints/seresnet50_rice_full_baseline/best_model.pth')
    parser.add_argument('--data-path', type=str, default='/home/master_core_ti/skripsi-grain/datasets/rice')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()

    evaluate(args.ckpt, args.data_path, split=args.split)
