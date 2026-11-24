import shutil
import tempfile
import time
from pathlib import Path

import torch


def bezpieczny_savefig(fig, sciezka, dpi=150, proby=5):
    # OneDrive czasem trzyma plik i savefig wywala Errno 22
    sciezka = Path(sciezka)
    sciezka.parent.mkdir(parents=True, exist_ok=True)
    last_err = None

    for i in range(proby):
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, dir=str(sciezka.parent)
            ) as f:
                tmp = Path(f.name)
            fig.savefig(tmp, dpi=dpi)
            shutil.move(str(tmp), str(sciezka))
            return
        except OSError as e:
            last_err = e
            if tmp is not None and tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            time.sleep(0.5 * (i + 1))

    tmp = Path(tempfile.gettempdir()) / (sciezka.stem + "_tmp.png")
    fig.savefig(tmp, dpi=dpi)
    try:
        shutil.copy2(tmp, sciezka)
    except OSError:
        alt = sciezka.with_name(sciezka.stem + f"_{int(time.time())}.png")
        shutil.copy2(tmp, alt)
        raise OSError(f"nie poszlo {sciezka}, jest {alt}") from last_err
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def bezpieczny_torch_save(obj, sciezka, proby=5):
    sciezka = Path(sciezka)
    sciezka.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for i in range(proby):
        tmp = sciezka.with_suffix(sciezka.suffix + f".tmp{i}")
        try:
            torch.save(obj, tmp)
            shutil.move(str(tmp), str(sciezka))
            return
        except OSError as e:
            last_err = e
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            time.sleep(0.5 * (i + 1))
    if last_err:
        raise last_err
