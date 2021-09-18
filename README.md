# Corn Seeds Classification

## Paper:  Automated Seed Quality Testing System using GAN & Active Learning
All the running code is in the src/ folder
To train your model, run main.py 
To use the Transfer Learning(pre_trained on Imagenet dataset) method, run the main_trLr_2.py ( to run this, you have to download the dataset from [here](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/sandeep_nagar_research_iiit_ac_in/Efqw-MBVMzVAhajCwpzWmqwBrNMK7zcREdr2ODMmycsd5w?e=ughRM6).

   Model-:        resnet18(pre_trained on Imagenet).
   
   Accuracy-:     86% 
   
   No.of Epochs-: 100.

## Dataset
   To download the dataset, click on this link [Dataset](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/sandeep_nagar_research_iiit_ac_in/EVXQD9ClwKtDvguuBsXefIgBexx27v2M8Ajhnwgl8-jixg?e=KklwXv)
which contains a zip file (data/train, data/test) 0.8 and 0.2, respectively  also the list of images names and their label(train_datafile.csv, test_datafile.csv) 
   Or create a folder to download the images run Seed_Classification/seeds_dataset/download_images.py, which will download images in the images folder.

   The dataset contains four classes: Discolored-0, Broken-2, Pure-1, Silkcut-3 (B, D, P, S).
   Now we can run the main.py 
   
    real images: 17802
    fake images: 20000(5K each class)
    new dataset: 57802
   
 

## Models
   Each image is fed into the input layer, which then talks to the next layer until the “output” layer is reached. The network’s “answer” comes from this final output layer.
   Well, we train networks by simply showing them many examples of what we want them to learn, hoping they extract the essence of the matter at hand and learn to ignore what doesn’t matter (like shape, size, colour or orientation).
   Loss and optimizer(Momentum and weight decay)
   
   Cross-Entropy Loss.
   
   Adam.
   
   SGD.

1. Resnet 18, 34, 50, 101, 152
2. MobileNetV2
3. Basic_model( 4 layers)
4. Squeezenet

   We explored large convolutional neural network models and trained for image classification in several ways.
   
   Choosing the classification model is difficult as we have a very small dataset and the distribution of the images of each class in the data set.
And applied some image augmentation techniques.
   We have used the Resnet(18, 34, 50, 101, 152), SqueezeNet, MobileNetV2 and simple model (3 CNN and 1 FC layers).
   
   Here the main problem is the overfitting, and to overcome this, we have used the Drop out method. 

Classification model (MobileNetV2)

   val_accuracy is 86%.
   
Classification model(resnet18)(updated)(e_200, optim-Adam,lr-0.001)
   
  Val acc is  86.03%.
   
## Training Details
   1. Traning the Cond. GAN (BigGAN).
   2. Traning the Batch Active Learning (BatchBALD).
   3. Training image classifier model.


   Each RGB image was preprocessed by resizing the smallest dimension to 128, cropping the centre 128x128 region, subtracting the per-pixel mean (across all images) and then using ten different sub-crops of size 128X128 (corners + centre with(out) horizontal flips). 

   Stochastic gradient descent with batch_size of 256 was used to update the parameters, starting with learning rate of 10−3, in conjunction with a momentum term of 0.1.

   We anneal the learning rate throughout training manually when the validation error plateaus.
## Classifying the sack of images after doing segmentation to get the single seed image.
   TODO
   
   
## Future  Work

   We can do more experiments by changing the  Learning rate.
   
   We are exploring more image augmentation techniques.
   
   I am visualizing the images after using image augmentation to know a more accurate effect of the same. 
   
   They are decreasing the difference between the train and test accuracy.
   
   More experiments on choosing the training model. 
   
   Improving overall accuracy. 
## requirements
pyTorch 
