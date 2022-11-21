import torchvision
from torchvision import transforms
from utils import plot_random_image
from data import downlaod_data, data_setup

downlaod_data(source='https://github.com/mrdbourke/pytorch-deep-learning/raw/46a8b2988295e135e39d3320b4dcf95fe2ef3927/data/pizza_steak_sushi.zip',
            destination='10_percent_food_pics')

downlaod_data(source='https://github.com/mrdbourke/pytorch-deep-learning/raw/46a8b2988295e135e39d3320b4dcf95fe2ef3927/data/pizza_steak_sushi_20_percent.zip',
            destination='20_percent_food_pics')
""
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights)

auto_transform = weights.transforms()

train_dir = '/home/ali/newdisk/Projects/my-project/PyTorch-tutorial/pipeline_for_create_and_experiment_models/data/10_percent_food_pics/train'
test_dir = '/home/ali/newdisk/Projects/my-project/PyTorch-tutorial/pipeline_for_create_and_experiment_models/data/10_percent_food_pics/test'

train_dataloader, test_dataloader, class_names = data_setup(train_dir=train_dir, 
                                                            test_dir=test_dir,
                                                            batch_size=32,
                                                            transform=auto_transform)

print(f"train_dataloader is: {train_dataloader}, test_dataloader is: {test_dataloader}, class_names is: {class_names}")

plot_random_image(model=model,
                    image_path=test_dir,
                    class_names=class_names,
                    transform=auto_transform,
                    n=3)
