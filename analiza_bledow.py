import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix

from przygotowanie_danych import ladna_nazwa

WYNIKI_DIR = Path("wyniki")
TOP_K_ZBLIZENIE = 10


def zbierz_predykcje(model, loader, device):
    model.eval()
    preds, pewnosc, etykiety = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            wyjscie = model(images.to(device))
            prob = torch.softmax(wyjscie, dim=1)
            p, pred = prob.max(dim=1)
            preds.extend(pred.cpu().tolist())
            pewnosc.extend(p.cpu().tolist())
            etykiety.extend(labels.tolist())
    return np.array(etykiety), np.array(preds), np.array(pewnosc)


def rysuj_pelna_cm(cm, nazwa):
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(nazwa + " - confusion matrix (znormalizowana po wierszach)")
    ax.set_xlabel("predykcja")
    ax.set_ylabel("prawdziwa rasa")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(WYNIKI_DIR / ("cm_" + nazwa + ".png"), dpi=150)
    plt.close(fig)


def najczesciej_mylone(cm, k):
    bez_diag = cm.copy()
    np.fill_diagonal(bez_diag, 0)
    wynik = bez_diag.sum(axis=1) + bez_diag.sum(axis=0)
    return list(np.argsort(-wynik)[:k])


def zblizenie_cm(cm, nazwy, nazwa, k=TOP_K_ZBLIZENIE):
    idx = najczesciej_mylone(cm, k)
    sub = cm[np.ix_(idx, idx)]
    # kolor - tylko pomyłki, poprawne nie są kolorowane
    frakcja = sub / cm[idx].sum(axis=1, keepdims=True)
    frakcja_bledy = frakcja.copy()
    np.fill_diagonal(frakcja_bledy, 0)
    etykiety = [nazwy[i] for i in idx]

    fig, ax = plt.subplots(figsize=(9, 8))
    vmax = max(float(frakcja_bledy.max()), 0.01)
    im = ax.imshow(frakcja_bledy, cmap="Reds", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(idx)))
    ax.set_yticks(range(len(idx)))
    ax.set_xticklabels(etykiety, rotation=45, ha="right")
    ax.set_yticklabels(etykiety)
    for i in range(len(idx)):
        for j in range(len(idx)):
            if sub[i, j] > 0:
                kolor = "white" if (i != j and frakcja_bledy[i, j] > 0.6 * vmax) else "black"
                ax.text(j, i, str(sub[i, j]), ha="center", va="center", color=kolor)
    ax.set_xlabel("predykcja")
    ax.set_ylabel("prawdziwa rasa")
    ax.set_title(nazwa + " - zblizenie: " + str(k) + " najczesciej mylonych ras")
    fig.colorbar(im, label="ulamek zdjec danej rasy (tylko pomylki)")
    fig.tight_layout()
    sciezka = WYNIKI_DIR / ("cm_zblizenie_" + nazwa + ".png")
    fig.savefig(sciezka, dpi=150)
    plt.close(fig)
    return sciezka, etykiety


def analiza_bledow(model, test_loader, id_to_class, device, nazwa, log):
    WYNIKI_DIR.mkdir(exist_ok=True)
    n = len(id_to_class)
    nazwy = [ladna_nazwa(id_to_class[i]) for i in range(n)]
    samples = test_loader.dataset.samples

    y, pred, pew = zbierz_predykcje(model, test_loader, device)

    # 1. confusion matrix
    cm = confusion_matrix(y, pred, labels=list(range(n)))
    np.savetxt(WYNIKI_DIR / ("cm_" + nazwa + ".csv"), cm, fmt="%d", delimiter=",")
    rysuj_pelna_cm(cm, nazwa)
    sciezka, etykiety = zblizenie_cm(cm, nazwy, nazwa)
    log.sekcja("ZBLIZENIE CM")
    log.pisz("rasy w zblizeniu:", ", ".join(etykiety))
    log.pisz("zapisano:", sciezka)

    # 2. najczestsze pomylki (para: prawdziwa -> przewidziana)
    bez_diag = cm.copy()
    np.fill_diagonal(bez_diag, 0)
    pary = []
    for i in range(n):
        for j in range(n):
            if bez_diag[i, j] > 0:
                pary.append((int(bez_diag[i, j]), nazwy[i], nazwy[j]))
    pary.sort(reverse=True)

    log.sekcja("NAJCZESTSZE POMYLKI")
    with open(WYNIKI_DIR / ("pomylki_" + nazwa + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["liczba", "prawdziwa", "przewidziana"])
        for ile, a, b in pary[:30]:
            w.writerow([ile, a, b])
    for ile, a, b in pary[:10]:
        log.pisz(ile, "x", a, "->", b)

    # 3. najsłabsze rasy (accuracy per klasa)
    acc_klasy = np.diag(cm) / cm.sum(axis=1)
    najgorsze = np.argsort(acc_klasy)[:10]
    log.sekcja("NAJSLABSZE RASY")
    for i in najgorsze:
        log.pisz(nazwy[i], "acc", round(float(acc_klasy[i]), 3), "n =", int(cm[i].sum()))

    # 4. przykłady błędnych predykcji - najpierw te, w których model był PEWNY
    bledy = [i for i in range(len(y)) if y[i] != pred[i]]
    bledy.sort(key=lambda i: -pew[i])
    wybrane = bledy[:12]

    fig, axes = plt.subplots(3, 4, figsize=(14, 11))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wybrane):
        ax.imshow(Image.open(samples[i][0]).convert("RGB"))
        ax.set_title("prawda: " + nazwy[y[i]] + "\npred: " + nazwy[pred[i]]
                     + " (" + str(round(float(pew[i]), 2)) + ")", fontsize=9)
    fig.tight_layout()
    fig.savefig(WYNIKI_DIR / ("bledy_" + nazwa + ".png"), dpi=150)
    plt.close(fig)

    log.pisz("bledow:", len(bledy), "/", len(y))
    log.pisz("zapisano: cm, zblizenie, pomylki, bledy dla", nazwa)
