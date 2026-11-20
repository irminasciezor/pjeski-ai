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


# np. n02085620-Chihuahua -> Chihuahua
def ladna_nazwa(folder):
    czesci = folder.split("-", 1)
    return czesci[-1]


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

    # preprocess pod ResNet (ImageNet)
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class DogsDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            path, label = self.samples[i]
            img = Image.open(path)
            img = img.convert("RGB")
            img = tfm(img)
            return img, label

    train_loader = DataLoader(DogsDataset(train_samples), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(DogsDataset(val_samples), batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(DogsDataset(test_samples), batch_size=BATCH, shuffle=False)

    return train_loader, val_loader, test_loader, class_to_id, id_to_class, num_classes


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_to_id, id_to_class, num_classes = przygotuj_dane()
    images, labels = next(iter(train_loader))
    print("przyklad batcha:", images.shape)
