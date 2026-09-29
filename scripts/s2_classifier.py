"""Mục 2 — Ứng dụng 1: fine-tune ResNet-18 trên TF Flowers, đánh giá một lần trên tập test."""
import json
import time
from pathlib import Path

from common import ART_DIR, DEVICE, FAST, FLOWERS_DIR, SEED, np, plt, save_fig, seed_everything, torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

import core.classifier as clf

OUT = ART_DIR / "classifier"


def loader(ds, shuffle: bool) -> DataLoader:
    # Windows: worker dùng spawn → dataset phải pickle được (ImageFolder + transform ở mức module)
    return DataLoader(ds, batch_size=64, shuffle=shuffle, num_workers=2, pin_memory=DEVICE == "cuda",
                      persistent_workers=True)


def run_epoch(model, dl, criterion, train: bool, optimizer=None, scheduler=None, scaler=None, use_amp=False):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in dl:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(DEVICE, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); scheduler.step()
        loss_sum += loss.item() * len(y); correct += (logits.argmax(1) == y).sum().item(); total += len(y)
    return loss_sum / total, correct / total


def main():
    seed_everything()
    base = ImageFolder(FLOWERS_DIR)
    classes, targets = base.classes, np.array(base.targets)
    all_idx = np.arange(len(targets))
    train_idx, tmp_idx = train_test_split(all_idx, test_size=0.2, stratify=targets, random_state=SEED)
    val_idx, test_idx = train_test_split(tmp_idx, test_size=0.5, stratify=targets[tmp_idx], random_state=SEED)
    if FAST:  # CPU: dùng 800 ảnh train để chạy trong vài phút
        train_idx = np.random.default_rng(SEED).choice(train_idx, size=min(800, len(train_idx)), replace=False)

    train_ds = Subset(ImageFolder(FLOWERS_DIR, transform=clf.TRAIN_TF), train_idx)
    val_ds = Subset(ImageFolder(FLOWERS_DIR, transform=clf.EVAL_TF), val_idx)
    test_ds = Subset(ImageFolder(FLOWERS_DIR, transform=clf.EVAL_TF), test_idx)
    train_dl, val_dl, test_dl = loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)

    # Lưu cách chia để mọi người tái lập đúng tập test
    (OUT / "split.json").write_text(json.dumps(
        {"train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()}))
    print(f"Lớp: {classes}\ntrain={len(train_ds)} · val={len(val_ds)} · test={len(test_ds)}")

    # Huấn luyện
    epochs = 1 if FAST else 5
    model = clf.build_model(len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=epochs * len(train_dl))
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_acc, history = 0.0, []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, True, optimizer, scheduler, scaler, use_amp)
        va_loss, va_acc = run_epoch(model, val_dl, criterion, False, use_amp=use_amp)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        if va_acc > best_acc:  # chọn checkpoint theo tập validation, không nhìn tập test
            best_acc = va_acc
            torch.save(model.state_dict(), OUT / "model.pt")
        print(f"epoch {epoch}/{epochs} · train loss {tr_loss:.3f} acc {tr_acc:.3f} · "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} · {time.time() - t0:.0f}s")
    (OUT / "classes.json").write_text(json.dumps(classes))
    print("Val accuracy tốt nhất:", round(best_acc, 4))

    # Đánh giá MỘT LẦN trên tập test với checkpoint tốt nhất
    model.load_state_dict(torch.load(OUT / "model.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for x, y in test_dl:
            y_pred += model(x.to(DEVICE)).argmax(1).cpu().tolist(); y_true += y.tolist()
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))
    metrics = {"test_accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
               "test_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
               "epochs": epochs, "history": history, "model": "resnet18-imagenet-finetune"}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=classes).plot(cmap="Blues", xticks_rotation=30)
    plt.title(f"Ma trận nhầm lẫn — test accuracy {metrics['test_accuracy']:.3f}")
    save_fig("classifier_confusion_matrix.png")

    # Dùng tầng suy luận giống hệt cách API sẽ dùng
    classifier = clf.ImageClassifier()
    sample_paths = [base.samples[i][0] for i in test_idx[:4]]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, p in zip(axes, sample_paths):
        t0 = time.perf_counter(); out = classifier.predict(Image.open(p)); ms = (time.perf_counter() - t0) * 1000
        top = out["predictions"][0]
        ax.imshow(Image.open(p)); ax.axis("off")
        ax.set_title(f"thật: {Path(p).parent.name}\ndự đoán: {top['label']} ({top['score']:.0%}) · {ms:.0f} ms")
    save_fig("classifier_samples.png")


if __name__ == "__main__":
    main()
