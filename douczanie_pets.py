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


# ResNet50 z Oxford Pets -> nasze psy
URL_WAG = "https://huggingface.co/flaviodell/oxford-pets-resnet50/resolve/main/best_model.pth"
WAGI_PETS = Path("wagi/oxford_pets_resnet50.pth")
WAGI_OUT = Path("wagi/douczanie_pets_dogs.pt")
WYNIKI_DIR = Path("wyniki")
KRZYWA_PNG = WYNIKI_DIR / "krzywa_douczanie_pets.png"
HISTORIA_CSV = WYNIKI_DIR / "historia_douczanie_pets.csv"
TYTUL = "douczanie pets"
NAZWA_LOG = "douczanie_pets"

EPOCHS = 15
LR = 0.001
EARLY_STOP_ACC = 0.83
PATIENCE = 5
KLAS_PETS = 37
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


def pobierz_wagi_pets(log):
    folder = Path("wagi")
    if not folder.exists():
        folder.mkdir()

    if WAGI_PETS.exists():
        log.pisz("mam juz wagi pets:", WAGI_PETS)
        return

    log.pisz("pobieram wagi pets z internetu...")
    # u mnie ssl czasem sie wywala, wiec tak:
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(URL_WAG, context=ctx) as odp:
        dane = odp.read()
    with open(WAGI_PETS, "wb") as f:
        f.write(dane)
    log.pisz("pobrane:", WAGI_PETS)


def zrob_model(num_classes, log):
    model = models.resnet50(weights=None)
    ile = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(ile, KLAS_PETS),
    )

    pobierz_wagi_pets(log)
    checkpoint = torch.load(WAGI_PETS, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    log.pisz("pets resnet50 wgrany (37 klas na starcie)")

    model.fc = nn.Linear(ile, num_classes)
    log.pisz("podmienilam glowe na", num_classes, "klas")
    return model


def zapisz_krzywa(historia, log):
    if not WYNIKI_DIR.exists():
        WYNIKI_DIR.mkdir()

    with open(HISTORIA_CSV, "w", newline="", encoding="utf-8") as f:
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
    ax1.set_title(TYTUL + " - accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epoki, train_loss, marker="o", color="tab:red", label="train loss")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.set_title(TYTUL + " - loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(KRZYWA_PNG, dpi=150)
    plt.close(fig)
    log.pisz("zapisano krzywa:", KRZYWA_PNG)
    log.pisz("zapisano historie:", HISTORIA_CSV)


def trenuj(model, train_loader, val_loader, device, log):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = -1.0
    best_wagi = None
    best_epoch = None
    bez_poprawy = 0
    historia = []
    ile_batchy = len(train_loader)

    log.pisz(
        "pets:", EPOCHS, "epok | lr", LR,
        "| stop gdy val >=", EARLY_STOP_ACC,
        "albo", PATIENCE, "epok bez lepszego val",
    )
    log.pisz("loader train:", ile_batchy, "batchy | wynik na dysk = best val")

    for epoch in range(EPOCHS):
        optimizer.param_groups[0]["lr"] = lr_cosine(LR, epoch, EPOCHS)
        t0 = time.time()
        log.pisz("[pets]", epoch + 1, "/", EPOCHS)
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
                    "  ~", i, "/", ile_batchy,
                    "loss", round(loss.item(), 4),
                    "acc biezaca", round(acc_teraz, 4),
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
            "po epoce", epoch + 1, ":",
            "loss", round(train_loss, 4),
            "train", round(train_acc, 4),
            "val", round(val_acc, 4),
            "|", format_czas(elapsed),
        )

        jest_best = False
        if val_acc > best_val:
            best_val = val_acc
            best_wagi = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            bez_poprawy = 0
            jest_best = True
            log.pisz("  ^ lepsze val, zapis modelu w pamieci:", best_epoch, round(best_val, 4))
        else:
            bez_poprawy = bez_poprawy + 1
            log.pisz("  val bez zmian", bez_poprawy, "/", PATIENCE)

        historia.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "elapsed_sec": round(elapsed, 2),
            "best": jest_best,
        })
        if val_acc >= EARLY_STOP_ACC:
            log.pisz("koniec treningu: val cel osiagniety")
            break

        if bez_poprawy >= PATIENCE:
            log.pisz("koniec treningu: patience")
            break

    return best_wagi, best_val, best_epoch, historia


def main():
    log = TreningLog(NAZWA_LOG)
    try:
        log.pisz("START - douczanie pets -> psy")
        random.seed(SEED)
        torch.manual_seed(SEED)

        device = ustaw_device(log)

        log.sekcja("DANE")
        train_loader, val_loader, test_loader, class_to_id, id_to_class, num_classes = przygotuj_dane()
        log.pisz("num_classes:", num_classes)
        log.pisz("train batches:", len(train_loader),
                 "val batches:", len(val_loader),
                 "test batches:", len(test_loader))

        log.sekcja("MODEL")
        model = zrob_model(num_classes, log)
        model = model.to(device)

        log.sekcja("TRENING")
        best_wagi, best_val, best_epoch, historia = trenuj(
            model, train_loader, val_loader, device, log
        )

        log.sekcja("ZAPIS WAG")
        torch.save(best_wagi, WAGI_OUT)
        log.pisz("zapisano NAJLEPSZE wagi (nie ostatnia epoka)")
        log.pisz("plik:", WAGI_OUT)
        log.pisz("best epoch:", best_epoch)
        log.pisz("best val:", best_val)
        log.pisz("ostatnia epoka w historii:", historia[-1]["epoch"] if historia else None)
        zapisz_krzywa(historia, log)
        log.pisz("KONIEC")
    finally:
        log.close()


if __name__ == "__main__":
    main()
