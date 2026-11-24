import copy
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb

import analiza_bledow
from analiza_bledow import analiza_bledow as uruchom_analize_bledow
from logowanie import TreningLog
from metryki import ewaluuj
from models import build_resnet18, build_resnet50, build_mobilenet
from przygotowanie_danych import przygotuj_dane, SEED
from scheduler import lr_cosine_czas
from gpu_utils import przygotuj_gpu
from io_utils import bezpieczny_savefig, bezpieczny_torch_save

AUGMENTACJA = True
LOAD_BEST = True
WEIGHTS_DIR = Path("wagi")
WYNIKI_DIR = Path("wyniki")
MODELS = ["resnet18", "resnet50", "mobilenet"]

LR = 1e-4  # przy 0.001 rozwalalo val na kontynuacji
EPOCHS = 500
EARLY_STOP_ACC = 1.01
PATIENCE = 10**9
PRINT_CO_BATCH = 20

CEL_CZAS_SEC = 2 * 60 * 60
# ile juz poszlo wczesniej (zeby dopelnic do 2h)
CZAS_JUZ_SEC = {
    "resnet18": 2371,
    "resnet50": 3279,
    "mobilenet": 1893,
}

WANDB_OFFLINE = True
WANDB_PROJECT = "pjeski-ai"
WANDB_WAGI = False


def max_czas_modelu(nazwa):
    juz = CZAS_JUZ_SEC.get(nazwa, 0)
    return max(60, CEL_CZAS_SEC - juz)


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


def zbuduj_model(name, num_classes, log):
    builders = {
        "resnet18": build_resnet18,
        "resnet50": build_resnet50,
        "mobilenet": build_mobilenet,
    }
    if name not in builders:
        log.pisz("nie znam modelu:", name)
        return None
    return builders[name](num_classes)


def zapisz_krzywa(historia, nazwa, log):
    if not WYNIKI_DIR.exists():
        WYNIKI_DIR.mkdir()

    krzywa_png = WYNIKI_DIR / ("krzywa_" + nazwa + ".png")
    historia_csv = WYNIKI_DIR / ("historia_" + nazwa + ".csv")

    with open(historia_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_acc",
                         "val_top3", "val_f1", "elapsed_sec", "best"])
        for wiersz in historia:
            writer.writerow([
                wiersz["epoch"],
                wiersz["lr"],
                wiersz["train_loss"],
                wiersz["train_acc"],
                wiersz["val_acc"],
                wiersz["val_top3"],
                wiersz["val_f1"],
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
    try:
        bezpieczny_savefig(fig, krzywa_png, dpi=150)
        log.pisz("zapisano krzywa:", krzywa_png)
    except OSError as e:
        log.pisz("png nie poszlo:", e)
    finally:
        plt.close(fig)
    log.pisz("zapisano historie:", historia_csv)


def trenuj(model, train_loader, val_loader, device, nazwa, log,
           max_czas_sec, best_val0=-1.0, best_wagi0=None):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = best_val0
    best_wagi = best_wagi0
    best_epoch = 0 if best_wagi0 is not None else None
    bez_poprawy = 0
    historia = []
    ile_batchy = len(train_loader)

    log.pisz("LR=", LR, "max_czas=", format_czas(max_czas_sec),
             "juz=", format_czas(CZAS_JUZ_SEC.get(nazwa, 0)),
             "batchy=", ile_batchy)

    t_start = time.time()

    for i in range(EPOCHS):
        if time.time() - t_start >= max_czas_sec:
            log.pisz("hardstop", format_czas(max_czas_sec), "przed epoka", i + 1)
            break

        lr = lr_cosine_czas(LR, time.time() - t_start, max_czas_sec)
        optimizer.param_groups[0]["lr"] = lr
        t0 = time.time()
        epoch_nr = i + 1
        log.pisz("===", nazwa, "epoch", epoch_nr,
                 "| lr", round(lr, 6),
                 "| czas", format_czas(time.time() - t_start), "/",
                 format_czas(max_czas_sec), "===")
        model.train()
        dobre = 0
        ile = 0
        suma_loss = 0.0
        hardstop = False

        for bi, (images, labels) in enumerate(train_loader, start=1):
            if time.time() - t_start >= max_czas_sec:
                log.pisz("stop - hardstop", format_czas(max_czas_sec),
                         "w epoce", epoch_nr, "batch", bi, "/", ile_batchy)
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

            if bi == 1 or bi % PRINT_CO_BATCH == 0 or bi == ile_batchy:
                acc_teraz = dobre / ile
                log.pisz(
                    "  batch", bi, "/", ile_batchy,
                    "loss", round(loss.item(), 4),
                    "acc", round(acc_teraz, 4),
                    "| razem", format_czas(time.time() - t_start),
                )

        if hardstop or ile == 0:
            break

        train_acc = dobre / ile
        train_loss = suma_loss / ile_batchy

        m = ewaluuj(model, val_loader, device)
        val_acc = m["acc"]
        elapsed = time.time() - t0
        log.pisz(
            "epoch", epoch_nr,
            "train_loss", round(train_loss, 4),
            "train", round(train_acc, 4),
            "val", round(val_acc, 4),
            "val_top3", round(m["top3"], 4),
            "val_f1", round(m["macro_f1"], 4),
            "czas", format_czas(elapsed),
            "| razem", format_czas(time.time() - t_start),
        )

        jest_best = False
        if val_acc > best_val:
            best_val = val_acc
            best_wagi = copy.deepcopy(model.state_dict())
            best_epoch = epoch_nr
            bez_poprawy = 0
            jest_best = True
            log.pisz("  best ep", best_epoch, "val", round(best_val, 4))
        else:
            bez_poprawy = bez_poprawy + 1
            log.pisz("  bez poprawy", bez_poprawy, "| best", round(best_val, 4))

        historia.append({
            "epoch": epoch_nr,
            "lr": lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_top3": m["top3"],
            "val_f1": m["macro_f1"],
            "elapsed_sec": round(elapsed, 2),
            "best": jest_best,
        })
        zapisz_krzywa(historia, nazwa, log)

        wandb.log({
            "epoch": epoch_nr,
            "lr": lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_top3": m["top3"],
            "val_macro_f1": m["macro_f1"],
        })

        if val_acc >= EARLY_STOP_ACC:
            log.pisz("stop - val >=", EARLY_STOP_ACC)
            break

        if bez_poprawy >= PATIENCE:
            log.pisz("stop - brak poprawy przez", PATIENCE, "epoki")
            break

        if time.time() - t_start >= max_czas_sec:
            log.pisz("stop - hardstop", format_czas(max_czas_sec),
                     "po epoce", epoch_nr)
            break

    log.pisz("czas calego modelu (ta sesja):", format_czas(time.time() - t_start))
    return best_wagi, best_val, best_epoch, historia


def main():
    WEIGHTS_DIR.mkdir(exist_ok=True)
    WYNIKI_DIR.mkdir(exist_ok=True)
    analiza_bledow.WYNIKI_DIR = WYNIKI_DIR

    sesja = TreningLog("run_sesja")
    try:
        sesja.pisz("START")
        sesja.pisz("tryb: augmentacja=", AUGMENTACJA, "| LOAD_BEST=", LOAD_BEST,
                   "| wagi:", WEIGHTS_DIR, "| wyniki:", WYNIKI_DIR)
        random.seed(SEED)
        torch.manual_seed(SEED)
        przygotuj_gpu(sesja)

        device = ustaw_device(sesja)

        sesja.sekcja("DANE")
        train_loader, val_loader, test_loader, class_to_id, id_to_class, num_classes = przygotuj_dane(
            augmentacja=AUGMENTACJA
        )
        sesja.pisz("num_classes:", num_classes)
        sesja.pisz("augmentacja:", AUGMENTACJA)
        sesja.pisz("train batches:", len(train_loader),
                   "val batches:", len(val_loader),
                   "test batches:", len(test_loader))
        sesja.pisz("modele do treningu:", ", ".join(MODELS))
        for n in MODELS:
            sesja.pisz(
                n, "| juz:", format_czas(CZAS_JUZ_SEC.get(n, 0)),
                "| teraz max:", format_czas(max_czas_modelu(n)),
                "| scheduler: cosine po czasie",
            )

        with open(WEIGHTS_DIR / "id_to_class.json", "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in id_to_class.items()}, f, ensure_ascii=False, indent=1)

        for nazwa in MODELS:
            log = TreningLog(nazwa)
            max_czas = max_czas_modelu(nazwa)
            run = wandb.init(
                project=WANDB_PROJECT,
                name=nazwa + "_z_aug_cont",
                mode="offline" if WANDB_OFFLINE else "online",
                config={
                    "model": nazwa,
                    "epochs_max": EPOCHS,
                    "lr": LR,
                    "scheduler": "cosine_czas",
                    "early_stop_acc": EARLY_STOP_ACC,
                    "patience": PATIENCE,
                    "max_czas_sec": max_czas,
                    "czas_juz_sec": CZAS_JUZ_SEC.get(nazwa, 0),
                    "seed": SEED,
                    "num_classes": num_classes,
                    "augmentacja": AUGMENTACJA,
                    "load_best": LOAD_BEST,
                },
            )
            try:
                log.pisz("--- MODEL:", nazwa, "---")
                log.pisz("device:", device)

                model = zbuduj_model(nazwa, num_classes, log)
                sciezka = WEIGHTS_DIR / (nazwa + ".pt")
                best_val0 = -1.0
                best_wagi0 = None

                if LOAD_BEST and sciezka.exists():
                    state = torch.load(sciezka, map_location="cpu", weights_only=True)
                    model.load_state_dict(state)
                    log.pisz("wczytano:", sciezka)
                    model = model.to(device)
                    m0 = ewaluuj(model, val_loader, device)
                    best_val0 = m0["acc"]
                    best_wagi0 = copy.deepcopy(model.state_dict())
                    log.pisz("start z dysku val", round(best_val0, 4),
                             "top3", round(m0["top3"], 4))
                else:
                    log.pisz("start od ImageNet")
                    model = model.to(device)

                log.sekcja("TRENING")
                best_wagi, best_val, best_epoch, historia = trenuj(
                    model, train_loader, val_loader, device, nazwa, log,
                    max_czas_sec=max_czas,
                    best_val0=best_val0,
                    best_wagi0=best_wagi0,
                )

                if best_wagi is None:
                    log.pisz("BRAK wag do zapisu - pomijam")
                    continue

                log.sekcja("ZAPIS WAG")
                bezpieczny_torch_save(best_wagi, sciezka)
                log.pisz("zapisano:", sciezka, "best ep", best_epoch, "val", best_val)
                if historia:
                    zapisz_krzywa(historia, nazwa, log)

                model.load_state_dict(best_wagi)
                log.sekcja("TEST")
                m = ewaluuj(model, test_loader, device)
                log.pisz("TEST acc:", round(m["acc"], 4),
                         "| top3:", round(m["top3"], 4),
                         "| macro-F1:", round(m["macro_f1"], 4))

                uruchom_analize_bledow(model, test_loader, id_to_class, device, nazwa, log)

                wandb.summary["best_epoch"] = best_epoch
                wandb.summary["best_val_acc"] = best_val
                wandb.summary["test_acc"] = m["acc"]
                wandb.summary["test_top3"] = m["top3"]
                wandb.summary["test_macro_f1"] = m["macro_f1"]
                wandb.log({
                    "krzywa": wandb.Image(str(WYNIKI_DIR / ("krzywa_" + nazwa + ".png"))),
                    "cm_pelna": wandb.Image(str(WYNIKI_DIR / ("cm_" + nazwa + ".png"))),
                    "cm_zblizenie": wandb.Image(str(WYNIKI_DIR / ("cm_zblizenie_" + nazwa + ".png"))),
                    "przyklady_bledow": wandb.Image(str(WYNIKI_DIR / ("bledy_" + nazwa + ".png"))),
                })
                if WANDB_WAGI:
                    art = wandb.Artifact("wagi-" + nazwa, type="model")
                    art.add_file(str(sciezka))
                    run.log_artifact(art)

                sesja.pisz(nazwa, "OK | best epoch", best_epoch,
                           "best val", best_val, "| TEST acc", round(m["acc"], 4),
                           "->", sciezka)

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            finally:
                run.finish()
                log.close()

        sesja.pisz("KONIEC")
    finally:
        sesja.close()


if __name__ == "__main__":
    main()
