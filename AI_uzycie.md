# Użycie AI

Korzystałyśmy z ChatGPT i Cursora jako wsparcia — do dopytywania wymagań, debugowania i weryfikacji pomysłów.  
Dobór modeli (ResNet-18/50, MobileNet), podział danych, early stop na val, cosine LR, augmentacja i porównanie transferu (Pets / Food-101) to nasze decyzje. AI nie pisało projektu „od zera”.

## Do czego konkretnie

- wyjaśnienie early stop: val vs train accuracy  
- czy cosine decay wystarczy jako „własny scheduler”  
- CUDA / cuDNN (crashe po długim treningu)  
- format wag z Hugging Face (`model_state` vs zwykły `state_dict`)  
- jak uczciwie porównać ImageNet vs Pets vs Food w raporcie  
- OneDrive blokujący zapis PNG (`Errno 22`)  
- kontynuacja treningu z checkpointu i limit czasu ~2h  

## Przykładowe prompty (realistyczne)

1. *„Mamy early stop na val_acc >= 0.83 i patience 5. Czy to jest early stopping na walidacji, czy wystarczy patrzeć na train acc? Projekt wymaga early stop.”*

2. *„Potrzebujemy własnego schedulera LR. Czy funkcja cosine decay w stylu lr = lr0 * 0.5 * (1 + cos(pi * epoch / epochs)) się liczy, czy muszą być osobna klasa i step()?”*

3. *„Ładuję wagi Oxford Pets z HF do ResNet50. W pliku jest `model_state`, a torchvision oczekuje state_dict. Jak to wczytać i potem podmienić głowę na 120 klas Stanford Dogs?”*

4. *„Trening leci na RTX 4050, po kilku godzinach wali `CUDNN_STATUS_EXECUTION_FAILED` / `BAD_PARAM_STREAM_MISMATCH` przy backward. Co zrobić żeby nocny run nie padał?”*

5. *„`fig.savefig('wyniki/krzywa_....png')` na OneDrive rzuca `OSError: [Errno 22] Invalid argument`. Jak zapisać wykres żeby trening się nie wywalił w połowie epoki?”*

6. *„Mamy już wagi po 15 epokach. Chcemy douczyć do łącznie 2h z tymi samymi wagami, bez startu od ImageNet. Cosine ma spadać w tej sesji. Jak to ustawić?”*


## Czego AI nie robiło

Nie wybierało architektury za nas, nie ustalało podziału 80/10/10 ani metryk do oddania.  
Kod i wyniki przechodziły przez nas — AI głównie do konsultacji i gaszenia pożarów (GPU, ścieżki, format checkpointów).
