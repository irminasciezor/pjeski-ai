from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


SEED = 2718
BATCH = 32
IMAGES_DIR = Path("data/Images")
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def ladna_nazwa(folder):
    # n02085620-Chihuahua -> Chihuahua
    return folder.split("-", 1)[-1]


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


def przygotuj_dane(augmentacja=True):
    random.seed(SEED)
    torch.manual_seed(SEED)

    # czasem jest data/Images/Images, nie pytaj
    folder = IMAGES_DIR
    if (IMAGES_DIR / "Images").exists():
        folder = IMAGES_DIR / "Images"

    print("biore zdjecia z:", folder)

    rasy = sorted(p for p in folder.iterdir() if p.is_dir())
    num_classes = len(rasy)
    print("liczba ras:", num_classes)

    class_to_id = {r.name: i for i, r in enumerate(rasy)}
    id_to_class = {i: r.name for i, r in enumerate(rasy)}

    samples = []
    for rasa in rasy:
        label = class_to_id[rasa.name]
        for plik in rasa.iterdir():
            if plik.suffix.lower() in (".jpg", ".jpeg"):
                samples.append((plik, label))

    print("liczba zdjec:", len(samples))

    etykiety = [s[1] for s in samples]
    # 80 / 10 / 10
    train_samples, tmp_samples = train_test_split(
        samples, test_size=0.2, stratify=etykiety, random_state=SEED,
    )
    etykiety_tmp = [s[1] for s in tmp_samples]
    val_samples, test_samples = train_test_split(
        tmp_samples, test_size=0.5, stratify=etykiety_tmp, random_state=SEED,
    )

    print("train:", len(train_samples), "val:", len(val_samples), "test:", len(test_samples))

    eval_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    if augmentacja:
        train_tfm = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])
        print("augmentacja TRAIN: TAK")
    else:
        train_tfm = eval_tfm
        print("augmentacja TRAIN: NIE")

    train_loader = DataLoader(DogsDataset(train_samples, train_tfm), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(DogsDataset(val_samples, eval_tfm), batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(DogsDataset(test_samples, eval_tfm), batch_size=BATCH, shuffle=False)

    return train_loader, val_loader, test_loader, class_to_id, id_to_class, num_classes


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_to_id, id_to_class, num_classes = przygotuj_dane()
    images, labels = next(iter(train_loader))
    print("przyklad batcha:", images.shape)
