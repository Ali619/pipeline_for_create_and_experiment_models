# A pipeline to train and compare 2 Torchvision models

In this pipeline, I choosed *EfficientNet_B0* and *EfficientNet_B2* to train them with 2 datasets and different amounts of epochs to see what is the best combination to get the most accuracy for my task which is a food classification. 

The instracture of using this pipline is in the `main_(notebook_version).ipynb` file. If you have any suggestion to upgrade this pipeline, feel free to open a **pull request** or let me know in the *discussions page*

**Note:** I used two datasets which is gathered from **food 101** dataset. One of them has *%10* and the other has *%20* of the images of *food 101* dataset. 
I did this to see how much the accuracy will get effect if we train the model with:
* different amount of data
* different size of model
* different train time *(epochs)*

The output of this pipeline will be shown in *tensorboard*, so we can see which model with which parameters have better accuracy.