import copy
import csv
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from logowanie import TreningLog
from models import build_resnet18, build_resnet50, build_mobilenet
from przygotowanie_danych import przygotuj_dane, SEED
from scheduler import lr_cosine

EPOCHS = 15
LR = 0.001
WEIGHTS_DIR = Path("wagi")
WYNIKI_DIR = Path("wyniki")
MODELS = ["resnet18", "resnet50", "mobilenet"]
EARLY_STOP_ACC = 0.83
PATIENCE = 5
PRINT_CO_BATCH = 20

def ustaw_device(log):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.pisz("device:", device)
    return device


def format_czas(sekundy):
    sekundy = int(round(sekundy))
    m = sekundy // 60
    s = sekundy % 60
    if m > 0:
        return str(m) + "m " + str(s) + "s"
    return str(s) + "s"


def zbuduj_model(name, num_classes, log):
    if name == "resnet18":
        return build_resnet18(num_classes)
    elif name == "resnet50":
        return build_resnet50(num_classes)
    elif name == "mobilenet":
        return build_mobilenet(num_classes)
    else:
        log.pisz("nie znam modelu:", name)
        return None


def zapisz_krzywa(historia, nazwa, log):
    if not WYNIKI_DIR.exists():
        WYNIKI_DIR.mkdir()

    krzywa_png = WYNIKI_DIR / ("krzywa_" + nazwa + ".png")
    historia_csv = WYNIKI_DIR / ("historia_" + nazwa + ".csv")

    with open(historia_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "elapsed_sec", "best"])
        for wiersz in historia:
            writer.writerow([
                wiersz["epoch"],
                wiersz["train_loss"],
                wiersz["train_acc"],
                wiersz["val_acc"],
                wiersz["elapsed_sec"],
                int(wiersz["best"]),
            ])

    epoki = [w["epoch"] for w in historia]
    train_acc = [w["train_acc"] for w in historia]
    val_acc = [w["val_acc"] for w in historia]
    train_loss = [w["train_loss"] for w in historia]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epoki, train_acc, marker="o", label="train acc")
    ax1.plot(epoki, val_acc, marker="o", label="val acc")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy")
    ax1.set_title(nazwa + " - accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epoki, train_loss, marker="o", color="tab:red", label="train loss")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.set_title(nazwa + " - loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(krzywa_png, dpi=150)
    plt.close(fig)
    log.pisz("zapisano krzywa:", krzywa_png)
    log.pisz("zapisano historie:", historia_csv)


def trenuj(model, train_loader, val_loader, device, nazwa, log):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = -1.0
    best_wagi = None
    best_epoch = None
    bez_poprawy = 0
    historia = []
    ile_batchy = len(train_loader)

    log.pisz("hiperparametry: EPOCHS=", EPOCHS, "LR=", LR,
             "EARLY_STOP_ACC=", EARLY_STOP_ACC, "PATIENCE=", PATIENCE)
    log.pisz("batchy na epoke:", ile_batchy)
    log.pisz("UWAGA: na dysk trafia NAJLEPSZY wynik (best val), NIE ostatnia epoka")

    for epoch in range(EPOCHS):
        optimizer.param_groups[0]["lr"] = lr_cosine(LR, epoch, EPOCHS)
        t0 = time.time()
        log.pisz("===", nazwa, "epoch", epoch + 1, "/", EPOCHS, "===")
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
                acc_teraz = dobre / ile
                log.pisz(
                    "  batch", i, "/", ile_batchy,
                    "loss", round(loss.item(), 4),
                    "acc", round(acc_teraz, 4),
                )

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
        log.pisz(
            "epoch", epoch + 1,
            "train_loss", round(train_loss, 4),
            "train", round(train_acc, 4),
            "val", round(val_acc, 4),
            "czas", format_czas(elapsed),
        )

        jest_best = False
        if val_acc > best_val:
            best_val = val_acc
            best_wagi = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            bez_poprawy = 0
            jest_best = True
            log.pisz("  NOWE BEST -> epoka", best_epoch, "val", round(best_val, 4))
        else:
            bez_poprawy = bez_poprawy + 1
            log.pisz("  brak poprawy,", bez_poprawy, "/", PATIENCE,
                     "| aktualne best: epoka", best_epoch, "val", round(best_val, 4))

        historia.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "elapsed_sec": round(elapsed, 2),
            "best": jest_best,
        })
        zapisz_krzywa(historia, nazwa, log)

        if val_acc >= EARLY_STOP_ACC:
            log.pisz("stop - val >=", EARLY_STOP_ACC)
            break

        if bez_poprawy >= PATIENCE:
            log.pisz("stop - brak poprawy przez", PATIENCE, "epoki")
            break

    return best_wagi, best_val, best_epoch, historia


def main():
    if not WEIGHTS_DIR.exists():
        WEIGHTS_DIR.mkdir()

    # wspolny log sesji + osobne logi per model
    sesja = TreningLog("run_sesja")
    try:
        sesja.pisz("START")
        random.seed(SEED)
        torch.manual_seed(SEED)

        device = ustaw_device(sesja)

        sesja.sekcja("DANE")
        train_loader, val_loader, test_loader, class_to_id, id_to_class, num_classes = przygotuj_dane()
        sesja.pisz("num_classes:", num_classes)
        sesja.pisz("train batches:", len(train_loader),
                   "val batches:", len(val_loader),
                   "test batches:", len(test_loader))
        sesja.pisz("modele do treningu:", ", ".join(MODELS))

        for nazwa in MODELS:
            log = TreningLog(nazwa)
            try:
                log.pisz("--- MODEL:", nazwa, "---")
                log.pisz("device:", device)

                model = zbuduj_model(nazwa, num_classes, log)
                model = model.to(device)

                log.sekcja("TRENING")
                best_wagi, best_val, best_epoch, historia = trenuj(
                    model, train_loader, val_loader, device, nazwa, log
                )

                sciezka = WEIGHTS_DIR / (nazwa + ".pt")
                log.sekcja("ZAPIS WAG")
                torch.save(best_wagi, sciezka)
                log.pisz("zapisano NAJLEPSZE wagi (nie ostatnia epoka)")
                log.pisz("plik:", sciezka)
                log.pisz("best epoch:", best_epoch)
                log.pisz("best val:", best_val)
                log.pisz("ostatnia epoka w historii:",
                         historia[-1]["epoch"] if historia else None)
                zapisz_krzywa(historia, nazwa, log)

                sesja.pisz(nazwa, "OK | best epoch", best_epoch,
                           "best val", best_val, "->", sciezka)

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            finally:
                log.close()

        sesja.pisz("KONIEC")
    finally:
        sesja.close()


if __name__ == "__main__":
    main()
