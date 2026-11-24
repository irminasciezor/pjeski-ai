from pathlib import Path
import csv

import torch

from przygotowanie_danych import przygotuj_dane
from models import build_resnet18, build_resnet50, build_mobilenet
from metryki import ewaluuj
from logowanie import TreningLog

WAGI_DIR = Path("wagi")
WYNIKI_DIR = Path("wyniki")
WYNIK_CSV = WYNIKI_DIR / "test_acc.csv"

MODELE = [
    ("resnet18", "resnet18.pt", "resnet18"),
    ("resnet50", "resnet50.pt", "resnet50"),
    ("mobilenet", "mobilenet.pt", "mobilenet"),
    ("douczanie_pets", "douczanie_pets_dogs.pt", "resnet50"),
    ("douczanie_jedzenie", "douczanie_jedzenie_dogs.pt", "resnet50"),
]


def zbuduj(typ, num_classes):
    if typ == "resnet18":
        return build_resnet18(num_classes)
    if typ == "resnet50":
        return build_resnet50(num_classes)
    if typ == "mobilenet":
        return build_mobilenet(num_classes)
    return None


def main():
    log = TreningLog("ewaluacja_test")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.pisz("device:", device)

        _, _, test_loader, _, _, num_classes = przygotuj_dane()
        log.pisz("klas:", num_classes, "test batches:", len(test_loader))
        WYNIKI_DIR.mkdir(exist_ok=True)

        wyniki = []
        for nazwa, plik, typ in MODELE:
            sciezka = WAGI_DIR / plik
            if not sciezka.exists():
                log.pisz("brak:", sciezka)
                continue

            log.sekcja(nazwa)
            model = zbuduj(typ, num_classes)
            model.load_state_dict(torch.load(sciezka, map_location=device, weights_only=True))
            model = model.to(device)

            m = ewaluuj(model, test_loader, device)
            log.pisz("test", round(m["acc"], 4), "top3", round(m["top3"], 4),
                     "f1", round(m["macro_f1"], 4))
            wyniki.append((nazwa, m["acc"], m["top3"], m["macro_f1"]))

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        with open(WYNIK_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "test_acc", "top3", "macro_f1"])
            for nazwa, acc, top3, f1 in wyniki:
                w.writerow([nazwa, round(acc, 4), round(top3, 4), round(f1, 4)])

        log.pisz("zapisano:", WYNIK_CSV)
        for nazwa, acc, top3, f1 in wyniki:
            log.pisz(nazwa, "->", round(acc, 4))
        log.pisz("KONIEC")
    finally:
        log.close()


if __name__ == "__main__":
    main()
