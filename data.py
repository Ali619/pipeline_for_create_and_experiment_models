from pathlib import Path
import requests
import zipfile
import os

def downlaod_data(source: str, destination: str, remove_source: bool=True) -> Path:
    
    data_path = Path('data')
    image_path = data_path / destination

    if image_path.is_dir():
        print(f"The {destination} path is already exict, skipping downlaod")
    else:
        image_path.mkdir(parents=True, exist_ok=True)

        target_file = Path(source).name

        with open(data_path / target_file, 'wb') as f:
            request = requests.get(source)
            print("downloading data from source")
            f.write(request.content)


        with zipfile.ZipFile(data_path / target_file, 'r') as zipref:
            print("Unziping data")
            zipref.extractall(image_path)

        if remove_source:
            os.remove(data_path / target_file)
    return image_path.absolute()

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

NUM_WORKER = os.cpu_count()

def data_setup(train_dir: str, test_dir: str, batch_size: int, transform: transforms.Compose=None, num_worker=NUM_WORKER):

    if transform:
        img_transform = transform

    else:
        img_transform = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])
                                    ])


    # Loading data with the label. This dataloader will use folder name as label
    train_data = datasets.ImageFolder(train_dir, transform=img_transform)
    test_data = datasets.ImageFolder(test_dir, transform=img_transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKER)
    test_dataloader = DataLoader(dataset= test_data, batch_size=batch_size, num_workers=NUM_WORKER)   

    return train_dataloader, test_dataloader, class_names