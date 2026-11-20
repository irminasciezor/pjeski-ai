# Użycie AI

W trakcie realizacji projektu korzystano z narzędzi AI (ChatGPT, Cursor) wspomagająco — głównie przy wyjaśnianiu wymagań, weryfikacji podejścia oraz diagnostyce błędów (m.in. konfiguracja GPU, format zapisanych wag). Decyzje projektowe oraz implementacja (dobór modeli, podział danych, early stopping na zbiorze walidacyjnym, harmonogram cosine dla learning rate) pozostawały po stronie autorów.

Przykładowe zagadnienia, w których AI służyło jako wsparcie merytoryczne:

- różnica między early stopping na zbiorze walidacyjnym a monitorowaniem accuracy na zbiorze treningowym
- czy cosine decay learning rate spełnia wymóg własnego schedulera, czy wymagana jest osobna klasa
- sprawdzenie dostępności CUDA w PyTorch oraz możliwość równoległego uruchomienia dwóch procesów treningowych
- różnice w strukturze checkpointów pobranych z Hugging Face (`model_state` vs bezpośredni `state_dict`)
- sposób uczciwego porównania w raporcie modeli startujących z ImageNet oraz z wag Oxford Pets / Food-101

Narzędzia AI nie były wykorzystywane do generowania całego projektu „od zera”, lecz do konsultacji szczegółów technicznych i treści raportu.
