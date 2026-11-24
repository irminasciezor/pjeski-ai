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
from metryki import ewaluuj
from scheduler import lr_cosine_czas
from gpu_utils import przygotuj_gpu
from io_utils import bezpieczny_savefig, bezpieczny_torch_save


# pets (37 klas) -> nasze 120 ras
URL_WAG = "https://huggingface.co/flaviodell/oxford-pets-resnet50/resolve/main/best_model.pth"
WAGI_PETS = Path("wagi/oxford_pets_resnet50.pth")
WAGI_OUT = Path("wagi/douczanie_pets_dogs.pt")
WYNIKI_DIR = Path("wyniki")
KRZYWA_PNG = WYNIKI_DIR / "krzywa_douczanie_pets.png"
HISTORIA_CSV = WYNIKI_DIR / "historia_douczanie_pets.csv"
TYTUL = "douczanie pets"
NAZWA_LOG = "douczanie_pets"

AUGMENTACJA = True
LOAD_BEST = True
EPOCHS = 500
LR = 1e-4
EARLY_STOP_ACC = 1.01
PATIENCE = 10**9
KLAS_PETS = 37
PRINT_CO_BATCH = 20

MAX_CZAS_SEC = 2 * 60 * 60
CEL_CZAS_SEC = MAX_CZAS_SEC
CZAS_JUZ_SEC = 0


def ustaw_device(log):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    log.pisz("pobieram pets...")
    # ssl u mnie czasem pada, stad to
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(URL_WAG, context=ctx) as odp:
        dane = odp.read()
    with open(WAGI_PETS, "wb") as f:
        f.write(dane)
    log.pisz("ok:", WAGI_PETS)


def zrob_model_od_pets(num_classes, log):
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


def zrob_model_kontynuacja(num_classes, log):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(WAGI_OUT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    log.pisz("wczytano kontynuacje z:", WAGI_OUT)
    return model


def zapisz_krzywa(historia, log):
    if not WYNIKI_DIR.exists():
        WYNIKI_DIR.mkdir()

    with open(HISTORIA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_acc", "elapsed_sec", "best"])
        for wiersz in historia:
            writer.writerow([
                wiersz["epoch"],
                wiersz["lr"],
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
    try:
        bezpieczny_savefig(fig, KRZYWA_PNG, dpi=150)
        log.pisz("zapisano krzywa:", KRZYWA_PNG)
    except OSError as e:
        log.pisz("png nie poszlo:", e)
    finally:
        plt.close(fig)
    log.pisz("zapisano historie:", HISTORIA_CSV)


def trenuj(model, train_loader, val_loader, device, log, best_val0=-1.0, best_wagi0=None):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = best_val0
    best_wagi = best_wagi0
    best_epoch = 0 if best_wagi0 is not None else None
    bez_poprawy = 0
    historia = []
    ile_batchy = len(train_loader)

    log.pisz("lr", LR, "max", format_czas(MAX_CZAS_SEC), "batchy", ile_batchy)

    t_start = time.time()

    for epoch in range(EPOCHS):
        if time.time() - t_start >= MAX_CZAS_SEC:
            log.pisz("stop - hardstop", format_czas(MAX_CZAS_SEC))
            break

        lr = lr_cosine_czas(LR, time.time() - t_start, MAX_CZAS_SEC)
        optimizer.param_groups[0]["lr"] = lr
        t0 = time.time()
        log.pisz("[pets]", epoch + 1, "| lr", round(lr, 6),
                 "|", format_czas(time.time() - t_start), "/", format_czas(MAX_CZAS_SEC))
        model.train()
        dobre = 0
        ile = 0
        suma_loss = 0.0
        hardstop = False

        for i, (images, labels) in enumerate(train_loader, start=1):
            if time.time() - t_start >= MAX_CZAS_SEC:
                log.pisz("stop - hardstop w epoce", epoch + 1, "batch", i)
                hardstop = True
                break

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

        if hardstop or ile == 0:
            break

        train_acc = dobre / ile
        train_loss = suma_loss / ile_batchy

        m = ewaluuj(model, val_loader, device)
        val_acc = m["acc"]
        elapsed = time.time() - t0
        log.pisz(
            "po epoce", epoch + 1, ":",
            "loss", round(train_loss, 4),
            "train", round(train_acc, 4),
            "val", round(val_acc, 4),
            "|", format_czas(elapsed),
            "| razem", format_czas(time.time() - t_start),
        )

        jest_best = False
        if val_acc > best_val:
            best_val = val_acc
            best_wagi = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            bez_poprawy = 0
            jest_best = True
            log.pisz("  ^ lepsze val:", best_epoch, round(best_val, 4))
        else:
            bez_poprawy = bez_poprawy + 1
            log.pisz("  val bez zmian", bez_poprawy, "/", PATIENCE)

        historia.append({
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "elapsed_sec": round(elapsed, 2),
            "best": jest_best,
        })
        try:
            zapisz_krzywa(historia, log)
        except OSError as e:
            log.pisz("zapis wynikow padl, lecimy dalej:", e)

        if val_acc >= EARLY_STOP_ACC:
            log.pisz("koniec treningu: val cel osiagniety")
            break

        if bez_poprawy >= PATIENCE:
            log.pisz("koniec treningu: patience")
            break

        if time.time() - t_start >= MAX_CZAS_SEC:
            log.pisz("stop - hardstop po epoce", epoch + 1)
            break

    log.pisz("czas calego modelu (ta sesja):", format_czas(time.time() - t_start))
    return best_wagi, best_val, best_epoch, historia


def main():
    log = TreningLog(NAZWA_LOG)
    try:
        log.pisz("START pets -> psy")
        random.seed(SEED)
        torch.manual_seed(SEED)
        przygotuj_gpu(log)

        device = ustaw_device(log)

        log.sekcja("DANE")
        train_loader, val_loader, test_loader, class_to_id, id_to_class, num_classes = przygotuj_dane(
            augmentacja=AUGMENTACJA
        )
        log.pisz("num_classes:", num_classes, "| augmentacja:", AUGMENTACJA)
        log.pisz("train batches:", len(train_loader),
                 "val batches:", len(val_loader),
                 "test batches:", len(test_loader))
        log.pisz("max czas tej sesji:", format_czas(MAX_CZAS_SEC))

        log.sekcja("MODEL")
        best_val0 = -1.0
        best_wagi0 = None
        if LOAD_BEST and WAGI_OUT.exists():
            model = zrob_model_kontynuacja(num_classes, log).to(device)
            m0 = ewaluuj(model, val_loader, device)
            best_val0 = m0["acc"]
            best_wagi0 = copy.deepcopy(model.state_dict())
            log.pisz("z dysku val", round(best_val0, 4), "top3", round(m0["top3"], 4))
        else:
            model = zrob_model_od_pets(num_classes, log).to(device)
            log.pisz("brak checkpointu, od Oxford Pets")

        log.sekcja("TRENING")
        best_wagi, best_val, best_epoch, historia = trenuj(
            model, train_loader, val_loader, device, log,
            best_val0=best_val0, best_wagi0=best_wagi0,
        )

        if best_wagi is None:
            log.pisz("BRAK wag do zapisu")
            return

        log.sekcja("ZAPIS WAG")
        bezpieczny_torch_save(best_wagi, WAGI_OUT)
        log.pisz("zapisano:", WAGI_OUT)
        log.pisz("best epoch (sesja):", best_epoch, "best val:", best_val)
        if historia:
            zapisz_krzywa(historia, log)

        log.sekcja("TEST")
        model.load_state_dict(best_wagi)
        m = ewaluuj(model, test_loader, device)
        log.pisz("test_acc:", round(m["acc"], 4),
                 "| top3:", round(m["top3"], 4),
                 "| macro-F1:", round(m["macro_f1"], 4))
        log.pisz("KONIEC")
    finally:
        log.close()


if __name__ == "__main__":
    main()
