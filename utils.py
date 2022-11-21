import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from pathlib import Path
from PIL import Image
from typing import Tuple
device = "cuda" if torch.cuda.is_available() else "cpu"

def plot_random_image(model: torch,
                        image_path: str,
                        class_names: list,
                        transform: torchvision.transforms = None,
                        image_size: Tuple[int, int]=(224, 224),
                        n: int=3,
                        device: torch.device=device):
    
    _image_path = list(Path(image_path).glob('*/*.jpg'))
    random_image_path = random.sample(population=_image_path, k=n)
    

    for images_path in random_image_path:
        img = Image.open(images_path)

        if transform:
            img_transform = transform
        elif not transform:
            img_transform = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        model.to(device)
        model.eval()
        with torch.inference_mode():

            transformed_image = img_transform(img).unsqueeze(dim=0)
            y_pred = model(transformed_image.to(device))
        
        # Get image prob and label
        image_prob = torch.softmax(y_pred, dim=1)
        image_label = torch.argmax(image_prob, dim=1)

        print(class_names, image_prob, image_label)
        # plt.figure()
        # plt.imshow(img)
        # plt.title(f"It's {class_names[image_label]}, Prob: {image_prob.max()*100}")
        # plt.axis(False)
        # plt.show()
