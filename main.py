import torch
from torch import nn
import torchvision
from torchvision import transforms
from engine import train
from utils import plot_random_image, model_summary, create_models, create_writer, save_model
from data import downlaod_data, data_setup
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_10_precent = downlaod_data(source='https://github.com/mrdbourke/pytorch-deep-learning/raw/46a8b2988295e135e39d3320b4dcf95fe2ef3927/data/pizza_steak_sushi.zip',
            destination='10_percent_food_pics')

data_20_precent = downlaod_data(source='https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip',
            destination='20_percent_food_pics')

train_dir_10percent = data_10_precent / 'train'
test_dir_10percent = data_10_precent / 'test'

train_dir_20percent = data_20_precent / 'train'
test_dir_20percent = data_20_precent / 'test'

BATCH_SIZE = 32

train_dataloader_10_percent, test_dataloader_10_percent, class_names = data_setup(train_dir=train_dir_10percent,
                                                                                    test_dir=test_dir_10percent,
                                                                                    batch_size=BATCH_SIZE,
                                                                                    num_worker=os.cpu_count())

train_dataloader_20_percent, test_dataloader_20_percent, class_names = data_setup(train_dir=train_dir_20percent,
                                                                                    test_dir=test_dir_20percent,
                                                                                    batch_size=BATCH_SIZE,
                                                                                    num_worker=os.cpu_count())

# We need to change the classifier output layer while we only have 3 class to predict (not 1000)
# We will do this in create_models functions
effnet_b0 = create_models(model_name='effnet_b0', num_classes=class_names)
effnet_b2 = create_models(model_name='effnet_b2', num_classes=class_names)

# Creating parameters to test 2 models with diffrent hyperparameters
num_epochs = [5, 10]
models = [effnet_b0, effnet_b2]
train_dataloaders = {"data_10_percent": [train_dataloader_10_percent],
                    "data_20_percent": [train_dataloader_20_percent]}

# Starting loop for training 
experiment_num = 1

for train_data_name, train_data in train_dataloaders.items():
    for model_name in models:
        for epochs in num_epochs:
            
            print(f"Experiment number is: {experiment_num}",
                    f"train_data is: {train_data_name}",
                    f"model_name is: {model_name}",
                    f"epoch number is: {epochs}")

            # Creating loss function and optimizer
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(params=model_name.parameters(), lr=0.001)

            result = train(model=model_name,
                            train_dataloader=train_data,
                            test_dataloader=test_dataloader_20_percent,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            device=device,
                            epochs=epochs,
                            writer=create_writer(experiment_name=train_data_name,
                                                    model_name=model_name,
                                                    extra=f"{epochs}_epochs"))

            save_filepath = f"07_{model_name}_{train_data_name}_{epochs}_epochs.pth"

            save_model(model=model_name,
                       target_dir="models",
                       model_name=save_filepath)

            experiment_num += 1