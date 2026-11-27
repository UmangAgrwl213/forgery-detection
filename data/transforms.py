import albumentations as A
from albumentations.pytorch import ToTensorV2


_NORMALIZE = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def get_train_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.15),
        A.RandomBrightnessContrast(p=0.25),
        _NORMALIZE,
        ToTensorV2(),
    ])


def get_val_transforms(image_size):
    return A.Compose([A.Resize(image_size, image_size), _NORMALIZE, ToTensorV2()])
