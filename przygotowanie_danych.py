from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


# stale
SEED = 2718
BATCH = 32
IMAGES_DIR = Path("data/Images")
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


# np. n02085620-Chihuahua -> Chihuahua
def ladna_nazwa(folder):
    czesci = folder.split("-", 1)
    return czesci[-1]


class DogsDataset(Dataset):
    def __init__(self, samples, tfm):
        self.samples = samples
        self.tfm = tfm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.tfm(img), label


def przygotuj_dane():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # czasem folder jest data/Images/Images
    folder = IMAGES_DIR
    if (IMAGES_DIR / "Images").exists():
        folder = IMAGES_DIR / "Images"

    print("biore zdjecia z:", folder)

    # lista folderow ras
    rasy = []
    for p in folder.iterdir():
        if p.is_dir():
            rasy.append(p)
    rasy = sorted(rasy)

    num_classes = len(rasy)
    print("liczba ras:", num_classes)

    # mapa nazwa <-> numer
    class_to_id = {}
    id_to_class = {}
    for i in range(len(rasy)):
        nazwa = rasy[i].name
        class_to_id[nazwa] = i
        id_to_class[i] = nazwa

    # wszystkie sciezki + etykiety
    samples = []
    for rasa in rasy:
        label = class_to_id[rasa.name]
        for plik in rasa.iterdir():
            koncowka = plik.suffix.lower()
            if koncowka == ".jpg" or koncowka == ".jpeg":
                samples.append((plik, label))

    print("liczba zdjec:", len(samples))

    # etykiety do stratify
    etykiety = []
    for s in samples:
        etykiety.append(s[1])

    # 80% train, potem polowa z reszty = val, polowa = test
    train_samples, tmp_samples = train_test_split(
        samples,
        test_size=0.2,
        stratify=etykiety,
        random_state=SEED,
    )

    etykiety_tmp = []
    for s in tmp_samples:
        etykiety_tmp.append(s[1])

    val_samples, test_samples = train_test_split(
        tmp_samples,
        test_size=0.5,
        stratify=etykiety_tmp,
        random_state=SEED,
    )

    print("train:", len(train_samples))
    print("val:", len(val_samples))
    print("test:", len(test_samples))

    # TRAIN: augmentacja (losowa)
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    # VAL / TEST: bez losowosci
    eval_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    train_loader = DataLoader(DogsDataset(train_samples, train_tfm), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(DogsDataset(val_samples, eval_tfm), batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(DogsDataset(test_samples, eval_tfm), batch_size=BATCH, shuffle=False)

    return train_loader, val_loader, test_loader, class_to_id, id_to_class, num_classes


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_to_id, id_to_class, num_classes = przygotuj_dane()
    images, labels = next(iter(train_loader))
    print("przyklad batcha:", images.shape)