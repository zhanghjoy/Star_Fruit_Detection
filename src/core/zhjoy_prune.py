# Ultralytics YOLOv12 🚀 Pruning + Finetune Script
# Supports L1, LAMP, GroupNorm pruning
'''
🚀 Default configuration to add in ultralytics/cfg/default.yaml:

# Pruning settings ---------------------------------------------------
prune:
  enable: False           # Enable/disable pruning
  method: 'LAMP'          # Pruning method: [LAMP, L1, GroupNorm,Random]
  target_speedup: 1.5     # Target FLOPs speedup ratio (>1 means faster)
  steps: 5                # Iterative pruning steps
  max_ratio: 0.6          # Maximum pruning ratio per layer
  finetune: True          # Whether to finetune after pruning
  finetune_epochs: 100    # Extra finetune epochs after pruning
  strict: False           # Strict shape matching when loading pruned weights

🚀 Example bash command:
python prune_finetune.py --model_path yolov12n.pt --target_speedup 1.5 --prune_method LAMP --finetune --finetune_epochs 50
'''


import sys, os, argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch_pruning as tp
from thop import profile, clever_format 
# U need to add/import the models which U wanna to skip here

# --- Suppress verbose output when counting FLOPs ---
class SilentOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._stdout


def build_pruner(opts, model, sample_input, ignored_layers=None):
    """
    Build pruner for YOLO model.
    """
    method = opts.method.lower()
    if method == "l1":
        importance = tp.importance.MagnitudeImportance(p=1)
        pruner_cls = tp.pruner.MagnitudePruner
    elif method == "lamp":
        importance = tp.importance.LAMPImportance(p=2)
        pruner_cls = tp.pruner.MagnitudePruner
    elif method == "groupnorm":
        importance = tp.importance.GroupNormImportance(p=2)
        pruner_cls = tp.pruner.GroupNormPruner
    elif method == "random":
        impprtance = tp.importance.RandomImportance()
         pruner_cls = tp.pruner.MagnitudePruner
    else:
        raise NotImplementedError("Only L1, LAMP, and GroupNorm are supported!")

    per_step_ratio = opts.max_ratio / max(opts.steps, 1)

    pruner = pruner_cls(
        model,
        sample_input,
        importance=importance,
        iterative_steps=opts.steps,
        pruning_ratio=per_step_ratio,
        max_pruning_ratio=opts.max_ratio,
        ignored_layers=ignored_layers or [],
        root_module_types=[nn.Conv2d, nn.Linear]
    )
    return importance, pruner


def run_pruning(model, pruner, example_input, target_speedup=1.5):
    """
    Iterative pruning loop with FLOPs/Params logging.
    """
    with SilentOutput():
        try:
            ori_flops, ori_params = tp.utils.count_ops_and_params(model, example_input)
        except:
            ori_flops, ori_params = profile(model, (example_input,))
    ori_flops_f, ori_params_f = clever_format([ori_flops, ori_params], "%.3f")
    print(f"[Before Pruning] FLOPs: {ori_flops_f}, Params: {ori_params_f}")

    step = 0
    while True:
        step += 1
        pruner.step(interactive=False)

        with SilentOutput():
            try:
                pruned_flops, pruned_params = tp.utils.count_ops_and_params(model, example_input)
            except:
                pruned_flops, pruned_params = profile(model, (example_input,))
        flops_f, params_f = clever_format([pruned_flops, pruned_params], "%.3f")
        speed = ori_flops / (pruned_flops + 1e-9)

        print(f"[Iter {step}] FLOPs: {ori_flops_f} -> {flops_f}, "
              f"Params: {ori_params_f} -> {params_f}, SpeedUp: {speed:.2f}")

        if speed >= target_speedup or pruner.current_step >= pruner.iterative_steps:
            break

    print("✅ Pruning Finished.")
    return model


def finetune(model, train_loader, epochs=5, lr=1e-3, device="cuda"):
    """
    Finetune the pruned model.
    Replace criterion with YOLO loss when integrated into YOLO training.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()  # ⚠️ Replace with YOLO loss in actual training

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Finetune Epoch {ep+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")
    print("🎯 Finetune Finished.")
    return model


def get_ignored_layers(model):
    """
    Keep EDH, MSEE, EUCB, AAttn modules unpruned.
    """
    ignored_layers = []
    for k, m in model.named_modules():
        if isinstance(m, EDH):
            ignored_layers.extend([m.cv2[0], m.cv2[1], m.cv2[2],
                                   m.cv3[0], m.cv3[1], m.cv3[2], m.dfl])
        if isinstance(m, MSEE):
            ignored_layers.append(m)
        if isinstance(m, EUCB):
            ignored_layers.append(m)
        if isinstance(m, AAttn):
            ignored_layers.append(m)
    return ignored_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="yolov12n.pt")
    parser.add_argument("--target_speedup", type=float, default=1.5)
    parser.add_argument("--prune_method", type=str, default="LAMP", choices=["LAMP", "L1", "GroupNorm"])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--max_ratio", type=float, default=0.6)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--finetune_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # --- Load YOLOv12 model ---
    model = torch.load(args.model_path, map_location=args.device)
    model.eval()

    # --- Dummy input ---
    example_input = torch.randn(1, 3, 640, 640).to(args.device)

    # --- Prepare ignored layers ---
    ignored_layers = get_ignored_layers(model)

    # --- Build pruner ---
    _, pruner = build_pruner(args, model, example_input, ignored_layers)

    # --- Run pruning ---
    model = run_pruning(model, pruner, example_input, target_speedup=args.target_speedup)

    # --- Finetune if enabled ---
    if args.finetune:
        # Replace with actual YOLO DataLoader
        from torch.utils.data import DataLoader
        train_loader = DataLoader([ (torch.randn(3,640,640), torch.randint(0,80,(1,))) for _ in range(10) ], batch_size=2)
        model = finetune(model, train_loader, epochs=args.finetune_epochs, device=args.device)

    # --- Save pruned model ---
    torch.save(model.state_dict(), "yolov12n_pruned.pth")
    print("✅ Saved pruned model to yolov12n_pruned.pth")


if __name__ == "__main__":
    main()

