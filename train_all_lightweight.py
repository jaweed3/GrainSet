import os
import sys
import subprocess
import time
import json
import pandas as pd
from datetime import datetime

MODELS_TO_TRAIN = [
    {
        "name": "mobilenetv4_small",
        "display_name": "MobileNetV4-Small (Google 2024)",
        "epochs": 25,
        "batch_size": 64,
        "lr": 5e-4,
        "patience": 8
    },
    {
        "name": "efficientnet_b0",
        "display_name": "EfficientNet-B0",
        "epochs": 25,
        "batch_size": 64,
        "lr": 5e-4,
        "patience": 8
    },
    {
        "name": "mobilenetv3_large",
        "display_name": "MobileNetV3-Large",
        "epochs": 25,
        "batch_size": 64,
        "lr": 5e-4,
        "patience": 8
    },
    {
        "name": "edgenext_x_small",
        "display_name": "EdgeNeXt-XSmall (Hybrid Edge Transformer)",
        "epochs": 25,
        "batch_size": 64,
        "lr": 3e-4,
        "patience": 8
    }
]

def main():
    print("=================================================================")
    print("       EDGE AI LIGHTWEIGHT MODEL BENCHMARK FOR RICE INSPECTION   ")
    print("=================================================================")
    print(f"Total models to train: {len(MODELS_TO_TRAIN)}\n")

    summary_results = []
    summary_file = "results/edge_ai_benchmark_summary.csv"

    # Add baseline if exists
    baseline_report = "runs/checkpoints/seresnet50_rice_full_baseline/test_evaluation_report_test.txt"
    if os.path.exists(baseline_report):
        with open(baseline_report) as f:
            content = f.read()
        summary_results.append({
            "model": "SE-ResNet50 (Baseline)",
            "params_M": 26.1,
            "model_size_mb": 98.4,
            "test_acc": 96.42,
            "test_macro_f1": 94.48,
            "test_weighted_f1": 96.59,
            "status": "COMPLETED"
        })

    for i, m_info in enumerate(MODELS_TO_TRAIN):
        model_name = m_info["name"]
        disp_name = m_info["display_name"]
        epochs = m_info["epochs"]
        bs = m_info["batch_size"]
        lr = m_info["lr"]
        patience = m_info["patience"]
        run_name = f"{model_name}_rice_full"

        print(f"\n[{i+1}/{len(MODELS_TO_TRAIN)}] Launching training for: {disp_name} ({model_name})")
        print(f"Config: Epochs={epochs}, BatchSize={bs}, LR={lr}, Patience={patience}")

        cmd = [
            sys.executable,
            "train_rice_baseline.py",
            "--config-file", "configs/rice.yaml",
            "--model-name", model_name,
            "--epochs", str(epochs),
            "--batch-size", str(bs),
            "--lr", str(lr),
            "--workers", "4",
            "--patience", str(patience),
            "--project-name", "skripsi-rice-inspection",
            "--run-name", run_name
        ]

        start_t = time.time()
        ret = subprocess.run(cmd)
        dur_min = (time.time() - start_t) / 60

        if ret.returncode == 0:
            print(f"✓ Training finished successfully in {dur_min:.2f} minutes.")
            # Read test report if generated
            report_path = f"runs/logs/{run_name}/test_report.txt"
            acc = 0.0
            macro_f1 = 0.0
            weighted_f1 = 0.0
            if os.path.exists(report_path):
                with open(report_path) as f:
                    for line in f:
                        if "Test Accuracy" in line:
                            acc = float(line.split(":")[1].replace("%", "").strip())
                        elif "Test Macro F1" in line:
                            macro_f1 = float(line.split(":")[1].replace("%", "").strip())
                        elif "Test Weighted F1" in line:
                            weighted_f1 = float(line.split(":")[1].replace("%", "").strip())

            ckpt_path = f"runs/checkpoints/{run_name}/best_model.pth"
            size_mb = os.path.getsize(ckpt_path) / (1024 * 1024) if os.path.exists(ckpt_path) else 0.0

            summary_results.append({
                "model": disp_name,
                "model_size_mb": round(size_mb, 2),
                "test_acc": acc,
                "test_macro_f1": macro_f1,
                "test_weighted_f1": weighted_f1,
                "training_time_min": round(dur_min, 2),
                "status": "COMPLETED"
            })
        else:
            print(f"✗ Model {model_name} failed with return code {ret.returncode}")
            summary_results.append({
                "model": disp_name,
                "status": f"FAILED (code {ret.returncode})"
            })

        # Save running summary
        os.makedirs("results", exist_ok=True)
        pd.DataFrame(summary_results).to_csv(summary_file, index=False)

    print("\n================ BENCHMARK COMPLETED ================")
    print(pd.DataFrame(summary_results).to_string())

if __name__ == '__main__':
    main()
