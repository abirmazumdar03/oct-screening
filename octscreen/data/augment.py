# octscreen/data/augment.py
from torchvision import transforms as T


def build_train_augment(aug):
    """Operates on the resized PIL grayscale image. Every knob comes from the config `aug:` block."""
    if aug is None or not aug.get("enabled", True):
        return None
    ops = []
    if aug.hflip_p > 0:
        ops.append(T.RandomHorizontalFlip(p=float(aug.hflip_p)))
    scale = tuple(float(s) for s in aug.scale)
    if aug.rotate_deg or aug.translate or scale != (1.0, 1.0):
        ops.append(T.RandomAffine(degrees=float(aug.rotate_deg),
                                  translate=(float(aug.translate), float(aug.translate)),
                                  scale=scale, fill=0))
    if aug.brightness or aug.contrast:
        ops.append(T.ColorJitter(brightness=float(aug.brightness), contrast=float(aug.contrast)))
    return T.Compose(ops) if ops else None
