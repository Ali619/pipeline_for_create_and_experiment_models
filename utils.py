import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from pathlib import Path
from PIL import Image
from typing import Tuple
device = "cuda" if torch.cuda.is_available() else "cpu"

def plot_random_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: list,
                        transform: torchvision.transforms = None,
                        image_size: Tuple[int, int]=(224, 224),
                        num_sample: int=3,
                        device: torch.device=device):
    """
    This function is for predicting class and creating plot with a trained model. The samples for predicting and 
    ploting will be choosen randomly.

    Args:
        model: the model we want to predict with.
        image_path: the path to our test image data (data that model didn't see yet)
        class_names: name of classes we have to be writen on plot title
        transform: type of transform we want to do with the images before predicting. (Check this from the model documention)
        image_size: size of image train dataset. Defualt is (224, 224)
        num_sample: number of sample we want to predict and plot.
        device: device to use for predicting (cpu or cuda). Defualt is set to `device` from `device = "cuda" if torch.cuda.is_available() else "cpu"`

    Example: 
            plot_random_image(model=model,
                    image_path=test_dir_20percent,
                    class_names=class_names,
                    transform=auto_transform,
                    num_sample=3)
    """
    _image_path = list(Path(image_path).glob('*/*.jpg'))
    random_image_path = random.sample(population=_image_path, k=num_sample)
    

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
        plt.figure()
        plt.imshow(img)
        plt.title(f"It's {class_names[image_label]}, Prob: {image_prob.max()*100:.2f}")
        plt.axis(False)
        plt.show()


from torchinfo import summary
def model_summary(model: torch.nn.Module, input_size: Tuple=(32, 3, 224, 224)):
    """
    This function will show us the model's layers and how the data shape will change in the model's layer.

    Args:
        `model`: the model we want to check the layers.
        `input_size`: the input size as a `tuple`. Default is `(32, 3, 224, 224)` while we are using the dataset,
         dataloader and EfficientNet models. You can change it to what you need.

    """
    return summary(model=model, 
            input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

from torch import nn
def create_models(model_name: str, num_classes: int) -> torch.nn.Module:
    """
    This function will craete a model which is only `effnet_b0` for EfficientNet_b0 or `effnet_b2` for EfficientNet_b2.
    While Efficient models has 1000 nodes in the linear layer (last layer / classify layer), we need to reduce it to the 
    number of our custom datasets classes. 

    Args:
        `model_name`: the model's namm we want to create.
        `num_classes`: the amount of classes that our data has (number of classes we want to predict)


    Example:
        create_models(model_name=`effnet_b0`, num_classes= `3`)
    """
    if model_name == 'effnet_b0':
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        model = torchvision.models.efficientnet_b0(weights=weights).to(device)
        model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=1280, out_features=num_classes))


    elif model_name == "effnet_b2":
        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        model = torchvision.models.efficientnet_b2(weights=weights).to(device)
        model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features=1408, out_features=num_classes))

    for param in model.features.parameters():
        param.requires_grad = False
    return model

from torch.utils.tensorboard import SummaryWriter
def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    return SummaryWriter(log_dir=log_dir)


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.
    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    torch.save(obj=model.state_dict(),
             f=model_save_path)