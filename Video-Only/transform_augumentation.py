from Config import CONFIG
from torchvision import transforms

def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Return torchvision transforms for the given split.

    Args:
        split : "train" | "val" | "test"
    """
    mean = [0.485, 0.456, 0.406]   # ImageNet stats (Swin pretrained)
    std  = [0.229, 0.224, 0.225]

    if split == "train" and CONFIG["use_augment"]:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomResizedCrop(
                CONFIG["face_size"], scale=(0.85, 1.0), ratio=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((CONFIG["face_size"], CONFIG["face_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])