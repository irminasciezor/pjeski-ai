import torch
from sklearn.metrics import f1_score


def ewaluuj(model, loader, device):
    model.eval()
    dobre = 0
    dobre_top3 = 0
    ile = 0
    wszystkie_pred = []
    wszystkie_etykiety = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            wyjscie = model(images)

            pred = wyjscie.argmax(dim=1)
            # [batch, 3]
            top3 = wyjscie.topk(3, dim=1).indices

            dobre += (pred == labels).sum().item()
            dobre_top3 += (top3 == labels.unsqueeze(1)).any(dim=1).sum().item()
            ile += labels.size(0)

            wszystkie_pred.extend(pred.cpu().tolist())
            wszystkie_etykiety.extend(labels.cpu().tolist())

    return {
        "acc": dobre / ile,
        "top3": dobre_top3 / ile,
        "macro_f1": f1_score(wszystkie_etykiety, wszystkie_pred, average="macro"),
    }