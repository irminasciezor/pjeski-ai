from pathlib import Path
import random
import copy
import urllib.request
import ssl
import csv
import time

import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt

from przygotowanie_danych import przygotuj_dane, SEED
from logowanie import TreningLog
from scheduler import lr_cosine


# food-101 resnet50, potem glowa pod psy
URL_WAG = "https://huggingface.co/VinnyVortex004/Food101-Classifier/resolve/main/food101_model.pth"
WAGI_FOOD = Path("wagi/jedzenie_food101_resnet50.pth")
WAGI_OUT = Path("wagi/douczanie_jedzenie_dogs.pt")
WYNIKI_DIR = Path("wyniki")

EPOCHS = 15
LR = 0.001
EARLY_STOP_ACC = 0.83
PATIENCE = 5
KLAS_FOOD = 101
PRINT_CO_BATCH = 20


def ustaw_device(log):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.pisz("food run na:", device, "(cuda dostepne:", torch.cuda.is_available(), ")")
    return device


def format_czas(sekundy):
    sekundy = int(round(sekundy))
    m = sekundy // 60
    s = sekundy % 60
    if m > 0:
        return str(m) + "m " + str(s) + "s"
    return str(s) + "s"


def pobierz_wagi_jedzenie(log):
    Path("wagi").mkdir(exist_ok=True)
    if WAGI_FOOD.exists():
        log.pisz("wagi food juz sa:", WAGI_FOOD)
        return

    log.pisz("sciagam food-101...")
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(URL_WAG, context=ctx) as odp:
        dane = odp.read()
    with open(WAGI_FOOD, "wb") as f:
        f.write(dane)
    log.pisz("ok, mam:", WAGI_FOOD)


def zrob_model(num_classes, log):
    model = models.resnet50(weights=None)
    ile = model.fc.in_features
    model.fc = nn.Linear(ile, KLAS_FOOD)

    pobierz_wagi_jedzenie(log)
    # ten plik z HF to od razu state_dict
    state = torch.load(WAGI_FOOD, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    log.pisz("zaladowalam food (101 klas)")

    model.fc = nn.Linear(ile, num_classes)
    log.pisz("glowa pod psy:", num_classes)
    return model


def zapisz_wyniki(historia, log):
    WYNIKI_DIR.mkdir(exist_ok=True)
    csv_path = WYNIKI_DIR / "historia_douczanie_jedzenie.csv"
    png_path = WYNIKI_DIR / "krzywa_douczanie_jedzenie.png"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "elapsed_sec", "best"])
        for w in historia:
            writer.writerow([
                w["epoch"], w["train_loss"], w["train_acc"],
                w["val_acc"], w["elapsed_sec"], int(w["best"]),
            ])

    epoki = [w["epoch"] for w in historia]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epoki, [w["train_acc"] for w in historia], marker="o", label="train")
    ax1.plot(epoki, [w["val_acc"] for w in historia], marker="o", label="val")
    ax1.set_title("jedzenie - acc")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epoki, [w["train_loss"] for w in historia], marker="o", color="tab:red")
    ax2.set_title("jedzenie - loss")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log.pisz("wyniki:", csv_path, "i", png_path)


def trenuj(model, train_loader, val_loader, device, log):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = -1.0
    best_wagi = None
    best_epoch = None
    bez_poprawy = 0
    historia = []
    ile_batchy = len(train_loader)

    log.pisz("food:", EPOCHS, "rund, lr", LR, "| early", EARLY_STOP_ACC, "lub", PATIENCE, "staly val")
    log.pisz("train loader:", ile_batchy, "batchy; po kazdej epoce aktualizuje wyniki/")

    for epoch in range(EPOCHS):
        optimizer.param_groups[0]["lr"] = lr_cosine(LR, epoch, EPOCHS)
        t0 = time.time()
        log.pisz("epoka", epoch + 1, "/", EPOCHS)
        model.train()
        dobre = 0
        ile = 0
        suma_loss = 0.0

        for i, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            wyjscie = model(images)
            loss = loss_fn(wyjscie, labels)
            loss.backward()
            optimizer.step()

            pred = wyjscie.argmax(dim=1)
            dobre = dobre + (pred == labels).sum().item()
            ile = ile + labels.size(0)
            suma_loss = suma_loss + loss.item()

            if i == 1 or i % PRINT_CO_BATCH == 0 or i == ile_batchy:
                log.pisz("  b", i, "/", ile_batchy,
                         "loss", round(loss.item(), 4),
                         "acc", round(dobre / ile, 4))

        train_acc = dobre / ile
        train_loss = suma_loss / ile_batchy

        model.eval()
        dobre = 0
        ile = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                pred = model(images).argmax(dim=1)
                dobre = dobre + (pred == labels).sum().item()
                ile = ile + labels.size(0)

        val_acc = dobre / ile
        elapsed = time.time() - t0
        log.pisz("-> train", round(train_acc, 4), "val", round(val_acc, 4),
                 "loss", round(train_loss, 4), format_czas(elapsed))

        jest_best = False
        if val_acc > best_val:
            best_val = val_acc
            best_wagi = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            bez_poprawy = 0
            jest_best = True
            log.pisz("  * best update", best_epoch, round(best_val, 4))
        else:
            bez_poprawy = bez_poprawy + 1
            log.pisz("  bez poprawy", bez_poprawy, "/", PATIENCE)

        historia.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "elapsed_sec": round(elapsed, 2),
            "best": jest_best,
        })
        # zapisuje po drodze, zeby nie stracic jak padnie
        zapisz_wyniki(historia, log)

        if val_acc >= EARLY_STOP_ACC:
            log.pisz("early stop (acc)")
            break
        if bez_poprawy >= PATIENCE:
            log.pisz("early stop (patience)")
            break

    return best_wagi, best_val, best_epoch, historia


def main():
    log = TreningLog("douczanie_jedzenie")
    log.pisz("start: food -> psy")
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = ustaw_device(log)

    train_loader, val_loader, test_loader, class_to_id, id_to_class, num_classes = przygotuj_dane()
    log.pisz("klas:", num_classes, "train/val/test batches:",
             len(train_loader), len(val_loader), len(test_loader))

    model = zrob_model(num_classes, log).to(device)
    best_wagi, best_val, best_epoch, historia = trenuj(
        model, train_loader, val_loader, device, log
    )

    torch.save(best_wagi, WAGI_OUT)
    log.pisz("zapisane:", WAGI_OUT, "best ep", best_epoch, "val", best_val)
    zapisz_wyniki(historia, log)
    log.pisz("koniec jedzenie")
    log.close()


if __name__ == "__main__":
    main()
