from pathlib import Path
from datetime import datetime


LOGI_DIR = Path("logi")


class TreningLog:
    def __init__(self, nazwa):
        if not LOGI_DIR.exists():
            LOGI_DIR.mkdir()

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.nazwa = nazwa
        self.path = LOGI_DIR / (nazwa + "_" + stamp + ".log")
        self._f = open(self.path, "w", encoding="utf-8")

        self.pisz("=" * 60)
        self.pisz("LOG MODELU:", nazwa)
        self.pisz("start:", datetime.now().isoformat(timespec="seconds"))
        self.pisz("plik logu:", self.path)
        self.pisz("=" * 60)

    def pisz(self, *args):
        tekst = " ".join(str(a) for a in args)
        linia = datetime.now().strftime("%H:%M:%S") + " | " + tekst
        print(tekst, flush=True)
        self._f.write(linia + "\n")
        self._f.flush()

    def sekcja(self, tytul):
        self.pisz("")
        self.pisz("---", tytul, "---")

    def close(self):
        self.pisz("=" * 60)
        self.pisz("koniec logu:", datetime.now().isoformat(timespec="seconds"))
        self.pisz("=" * 60)
        self._f.close()
