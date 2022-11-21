import torchvision
from torchvision import transforms
# from torchinfo
from utils import plot_random_image
from data import downlaod_data, data_setup

data_10_precent = downlaod_data(source='https://github.com/mrdbourke/pytorch-deep-learning/raw/46a8b2988295e135e39d3320b4dcf95fe2ef3927/data/pizza_steak_sushi.zip',
            destination='10_percent_food_pics')

data_20_precent = downlaod_data(source='https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip',
            destination='20_percent_food_pics')

train_dir_10percent = data_10_precent / 'train'
test_dir_10percent = data_10_precent / 'test'

train_dir_20percent = data_20_precent / 'train'
test_dir_20percent = data_20_precent / 'test'

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights)

auto_transform = weights.transforms()


train_dataloader, test_dataloader, class_names = data_setup(train_dir=train_dir_10percent, 
                                                            test_dir=test_dir_10percent,
                                                            batch_size=32,
                                                            transform=auto_transform)

print(f"train_dataloader is: {train_dataloader}, test_dataloader is: {test_dataloader}, class_names is: {class_names}")

plot_random_image(model=model,
                    image_path=test_dir_20percent,
                    class_names=class_names,
                    transform=auto_transform,
                    n=3)
