import gc

import torch


def przygotuj_gpu(log=None, wylacz_cudnn=True):
    # cudnn sypal sie po nocnych runach, bez niego wolniej ale mniej crashy
    torch.backends.cudnn.enabled = not wylacz_cudnn
    torch.backends.cudnn.benchmark = False

    if not torch.cuda.is_available():
        if log:
            log.pisz("brak CUDA -> CPU")
        return

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    x = torch.randn(2, 3, 64, 64, device="cuda")
    y = torch.nn.Conv2d(3, 8, 3, padding=1).cuda()(x)
    y.sum().backward()
    del x, y
    torch.cuda.empty_cache()

    if log:
        log.pisz("GPU:", torch.cuda.get_device_name(0),
                 "cudnn=", torch.backends.cudnn.enabled)
