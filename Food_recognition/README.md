# Food recognition
In this project we explore different models that can be used in an image recognition task. In particular, the problem we are going to solve is to predict if the food in the image is 
sashimi or spaghetti. Finally, we present some visualization techniques to interpret what is happening to an image when it goes through the convolutional layers. 

<b>Problem:</b> image recognitin <br>
<b>Target classes:</b> sashimi and spaghetti_bolognese <br>
<b>Data source</b>: two classes of  ... <br>

## Content of the Repository

- <b> Custom_functions </b> This folder contains the custom functions used for plots and a custom class used for creating the folders needed for the training. 
- <b> Data </b> This folder contains a folder with the original data and the folders with images used for the training of the model.
- <b> Model </b> This folder we have saved all the models trained in the notebook along with the history of the trainng. 
- <b> Food_recognition.ipynb </b> This is the notebook used for training the models and explore the convolutional layers.

## Structure of the Code

### Modeling 
- <b> First Model </b> We start implementig a simple model where the images will go through some convolutional layers and then a neural network with just one hidden layer. <br>
- <b> Second Model </b> We use data augmentation and a dropout layer in order to mitigate overfitting. <br>
- <b> Third Model </b> We use a pretrained convolutional network (VGG16) to improve even more the performance of our model. In this step we will first extract the features of all the 
  images used the convolutional layers of VGG16 then we wil use them to train the fully connect classifier. <br>
- <b> Fourth Model </b> We then use the data augmentation.
- <b> Fifth Model </b> Finally, we use the technique of fine tuning and we will traine the convolutional layers of the VGG16. 

### Visualization 

- <b> Visualize of intermediate activation </b> We plot a 2D image for every channel of the maps
- <b> Visualization of the filters </b> We look at which patterns are captured by the filters
- <b> Heatmap of the class activation </b> We look at which part of an image where more involved in the prediction of the class. 
