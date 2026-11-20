# pjeski-ai

Klasyfikacja ras psów na zbiorze Stanford Dogs — ResNet50, transfer learning, early stop i cosine decay na learning rate.

## Uruchomienie

Potrzebujesz folderu `data/Images` (obrazy Stanford Dogs) oraz venv z PyTorch i torchvision.

```text
python przygotowanie_danych.py
python run.py
python douczanie_pets.py
python douczanie_jedzenie.py
```

`run.py` to główna ścieżka (od zera). Skrypty `douczanie_*` uczą od wag pobranych z internetu: Oxford Pets albo Food-101, potem podmiana głowy na nasze klasy.

**Gdzie co ląduje:** wagi w `wagi/`, tekstowe logi w `logi/`, CSV i wykresy w `wyniki/`.

Szczegóły harmonogramu LR: `scheduler.py`. Jak korzystaliśmy z AI przy pisaniu kodu: `AI_uzycie.md`.

PoC na ~10 rasach jest w notebookach `POC_*.ipynb`.

## Autorzy

- Agnieszka P (64682) — dsw64682@student.ideis.pl  
- Irmina S (64165) — dsw64165@student.ideis.pl