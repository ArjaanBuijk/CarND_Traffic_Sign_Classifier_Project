# Traffic Sign Recognition using a Convolutional Neural Network

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---
### Files

Link to my [project code (Jupyter Notebook)](https://github.com/ArjaanBuijk/CarND_Traffic_Sign_Classifier_Project/blob/master/Traffic_Sign_Classifier.ipynb)

Link to my [project code (HTML exported after run)](https://github.com/ArjaanBuijk/CarND_Traffic_Sign_Classifier_Project/blob/master/Traffic_Sign_Classifier.html)
  
 NOTE: You can not view HTML file directly on github, because it is too big. You must download it to your disk, and then open it in your browser to view it.

### Imports

#### All import statements are placed at the top of the notebook.

The code for this step is contained in the first code cell of the Jupyter notebook. 


### Hyper Parameters

#### All hyper parameters are defined as global variables, and defined in a single code cell, to allow easy optimization studies.

The code for this step is contained in the second code cell of the Jupyter notebook. 

The following hyper parameters are available for optimization:

| Hyper Parameter | Description | Optimized Value |
| --------------- | ----------- | --------------- |
| MAX_ANGLE | Max rotation angle during augmentation (see below) | 20 |
| CLIP_LIMIT | clip limit during CLAHE (see below) | 0.1 |
| EPOCHS | how many forward/backward loops | 200 |
| BATCH_SIZE| training images per batch | 16 |
| KEEP_PROB| dropout rate | 0.5 |
| RATE | learning rate | 0.0001 |

 I ran many variations of these parameters, and ended up with the optimized values given in above table.  Most surprising to me was the fact that the outcome is so sensitive to the BATCH_SIZE. 

### Loading data

#### All data was downloaded, stored in a folder and loaded from disk into numpy arrays.

The code for this step is contained in the third code cell of the Jupyter notebook. 

The downloaded pickle files are stored in the folder: ./data
It was already sub-divided into a training, validation and test data set.

The images (X) and the labels (y) are read into these numpy arrays:

 - X_train , y_train
 - X_valid, y _valid
 - X_test, y_test
 
These are the training, validation and test sets respectively.
  

### Data Set Summary & Exploration

#### 1. The data sets were first investigated for some basic information.

The code for this step is contained in the fourth & fifth code cell of the Jupyter notebook.  

The function check_and_summarize_data provides this summary:

| Item | Value |
| ---- | ----- |
|Number of training examples | 34799 | 
|Number of validation examples | 4410 |
|Number of testing examples | 12630 |
|Image data shape | (32, 32, 3) |
|Number of unique classes in training examples | 43 |
|Number of unique classes in validation examples | 43 |
|Number of unique classes in testing examples | 43 |
|Number of unique classes in all | 43 |

Key take-aways are:

 - There are 43 classes
 - There are 34799 images in the training set
 - The images are in RGB (3 channels)

 

#### 2. More in depth, visual investigation of the data.

The code for this step is contained in the sixth and seventh code cell of the Jupyter notebook.  

I start by plotting a class distribution histogram, visualizing how many images for each class are in the training and test data. This shows that:

 1. there is a large difference between number of items in each class. 
 2. the ratio of test data vs training data is consistent.

![class_distribution_histogram](https://github.com/ArjaanBuijk/CarND_Traffic_Sign_Classifier_Project/blob/master/class_distribution_histogram.jpg)
 
The description belonging to each class is read into a Panda DataFrame, from the file signnames.csv.
Instead of just referring to class labels, the description is mostly used in investigative reports.

In code cell six are also two functions for image manipulation that are used during this initial investigation and below during image pre-processing.

| Function | Description |
| -------- | ----------- |
|grayscale | converts an RGB image into grayscale | 
|apply_clahe | Applies a Contrast Limited Adaptive Histogram Equalization (CLAHE)|

The CLAHE technique was found during internet research on image processing to get more contrast in the pictures, and the method used is from the [scikit.exposure](http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist) library.

During training, it was found that applying the CLAHE technique was very beneficial to get higher accuracy.

A final review of the images is done by means of a table that summarizes for each class:

 - The class label
 - The description of the class
 - The number of images in the training set for this class
 - The RGB image of the first item in the training set 
 - The first image after Grayscale
 - The first image after Grayscale and CLAHE
 
![Table of Training Images](https://github.com/ArjaanBuijk/CarND_Traffic_Sign_Classifier_Project/blob/master/signs_training_summary.jpg)

### Design and Test a Model Architecture

#### 1. The training data is pre-processed using augmentation, Grayscale conversion and Contrast Limited Adaptive Histogram Equalization (CLAHE)

The code for this step is contained in the eight to eleventh code cells of the Jupyter notebook, while re-using functions defined earlier.

During detailed investigation of the images, as described above, I found 3 issues with the training set:

 1. There was a large difference in number of images per class.
 2. The traffic signs do NOT require color to be taken into account for classification.
 3. Even after conversion to Grayscale, there were a lot of images with very poor contrast.

These issues were addressed by the following 3 pre-processing techniques:

 1. I augmented the training data set, and added images to each class by taking the existing images and applying a random rotation. After this step, each class ended up with the same number of images. The total number of images increased from 34799 to 86430.
 2. I converted all training, validation and test images to Grayscale.
 3. After that, I applied CLAHE, which is both an equalizer and a normalization.

The effect of Grayscale and CLAHE was already described and shown in the previous section. 

At the end of the pre-processing steps, the data is written to a pickle file. Especially the augmentation with CLAHE is time consuming, and it is important to avoid these during hyper-parameter optimization.

#### 2. The model architecture and utility functions.

The code for my final model is located in the twelfth to sixteenth cells of the Jupyter notebook. 

<u>Description of LeNet-5 function:</u>

Because the model showed a tendency to slight over fitting during training, I added a dropout layer. In all, I made the following 3 modifications to the LeNet-5 implementation:

 1. Added a dropout layer before the last readout layer.
 2. Renamed the layers and did not reuse the same name twice.
 3. To allow visualization of layer activation, the layer identifier must be known outside of the function. This is why I am returning not just the logits, but also the identifiers of the hidden layers from the function, I guess that means these layers are no longer hidden ;-) 

My final model consisted of the following layers:

| Layer | Name | Description | 
| ----- | ---- | ----------- | 
| Input   |      		| 32x32x1 Grayscale image   							| 
| Convolution 5x5|conv1 | 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU		|conv1_r|												|
| Max pooling|conv1_p	| 2x2 stride,  VALID padding, outputs 14x14x6 |
| Convolution 5x5|conv2	  | 1x1 stride, VALID padding, outputs 10x10x16  |
| RELU	|conv2_r	|												|  
| Max pooling|conv2_p		| 2x2 stride, VALID padding, outputs 5x5x16 |
| Flatten|fc0		| outputs 400 |    						
| Fully connected |fc1	| outputs 120	|
| RELU	|fc1_r	|												|
| Fully connected |fc2	| outputs 84|
| RELU	|fc2_r	|
| Dropout |fc2_drop	|  |
| Output layer: Fully connected |logits | outputs 43|

In addition, there are several utility functions to monitor and evaluate the training. These utility functions are critical during hyper parameter optimization, to figure out why it is not converging properly.

<u>Description of utility functions:</u>

| function | Code Cell | Description |
| -------- | --------- | ----------- |
|evaluate|15|Evaluate the accuracy of predictions|
|prediction_counts|16|Count number of true-positive, false-negative, false-positive predictions for each class|
|summarize_predictions|16|Summarizes predictions in tabular and visual formats|
|summarize_top_probabilities|16|Summarizes top softmax probabilities |
|plot_accuracies_vs_epochs|16|Plots evolution of training & validation accuracies|
 
 
#### 4. Training approach.

The code for loading the pre-processed data is contained in the seventeenth cell of the Jupyter notebook. 
The training session is defined in the eigthteenth cell of the Jupyter notebook.

To train the model, I used the AdamOptimizer. This is an optimizer that automatically adapts its learning rate, includes bias-correction and momentum. It is recommended as the best overall optimizer in most cases [[ref](http://sebastianruder.com/optimizing-gradient-descent/index.html#whichoptimizertochoose)].

One interesting comment is made in the TensorFlow documentation about the epsilon parameter [[ref](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)]:

*The default value of 1e-8 for epsilon might not be a good default in general. For example, when training an Inception network on ImageNet a current good choice is 1.0 or 0.1.*

Because of this comment, I did a study if for this data set the default should be changed, and came to the conclusion that it does not help improve the end result. 

To determine the best hyper-parameters, I simply ran many variations in a systematic manner. Varying one parameter at a time. Each time, I looked at the evolution of training and validation accuracy to determine if the convergence is good, and if there is over fitting or under fitting. The final parameters chosen gave this evolution:

![convergence](https://github.com/ArjaanBuijk/CarND_Traffic_Sign_Classifier_Project/blob/master/convergence_final.jpg)

In addition, I found it helpful to review a table of the predictions of true positives, false negatives and false positives for each class. Both the total numbers and images for 2 of the false negatives for each class.

For example, this allowed me to spot that I was initially using a too large an angle during rotation. Right direction only signs were false identified as left direction only sign. After limiting the angle during image rotation to 20 degrees, those signs no longer showed up in false negatives of the validation set.

The evaluation of the validation results for the final model (see table below) shows that:
 - The predictor is good at detecting a speed limit sign, but sometimes missed the speed limit value. For example, it predicted 70 km/h instead 20 km/h. 
 - Several 'danger'signs, like slippery road, curve to right, curve to left, double curve, etc.. were wrongly predicted. 

Running the classifier on the test set gave an accuracy of <b>0.952</b>

<u>Detailed prediction summary:</u>
 
 To show the impact of augmenting the data, I am here listing the results of the trained classifier without data augmentation and with data augmentation. 

| accuracy measure | without data augmentation | with data augmentation |
| ---------------- | ------------------------- | -------------------------- |
| training accuracy | 1.0 | 1.0 |
| validation accuracy | 0.955 |  0.970 |
| test accuracy | 0.932 | 0.950 |

The details of validation prediction, when using the final trained model using data augmentation on the training set, is given in this table:

| label | count | true-pos | false-neg | false-pos |
| ----- | ----- | -------- | --------- | --------- |
|class =0: Speed limit (20km/h)                                   | 30 |   25(83 %)  |   5(16 %)   |      1  |
|class =1: Speed limit (30km/h)                                   |240 |  237(98 %)  |   3(1  %)   |      6  |
|class =2: Speed limit (50km/h)                                   |240 |  225(93 %)  |  15(6  %)   |      2  |
|class =3: Speed limit (60km/h)                                   |150 |  137(91 %)  |  13(8  %)   |      6  |
|class =4: Speed limit (70km/h)                                   |210 |  207(98 %)  |   3(1  %)   |      6  |
|class =5: Speed limit (80km/h)                                   |210 |  208(99 %)  |   2(0  %)   |     17  |
|class =6: End of speed limit (80km/h)                            | 60 |   60(100%)  |   0(0  %)   |      0  |
|class =7: Speed limit (100km/h)                                  |150 |  148(98 %)  |   2(1  %)   |      2  |
|class =8: Speed limit (120km/h)                                  |150 |  148(98 %)  |   2(1  %)   |      6  |
|class =9: No passing                                             |150 |  144(96 %)  |   6(4  %)   |      0  |
|class =10: No passing for vehicles over 3.5 metric tons          |210 |  210(100%)  |   0(0  %)   |      5  |
|class =11: Right-of-way at the next intersection                 |150 |  150(100%)  |   0(0  %)   |      4  |
|class =12: Priority road                                         |210 |  210(100%)  |   0(0  %)   |      3  |
|class =13: Yield                                                 |240 |  239(99 %)  |   1(0  %)   |      0  |
|class =14: Stop                                                  | 90 |   83(92 %)  |   7(7  %)   |      0  |
|class =15: No vehicles                                           | 90 |   90(100%)  |   0(0  %)   |      1  |
|class =16: Vehicles over 3.5 metric tons prohibited              | 60 |   33(55 %)  |  27(45 %)   |      1  |
|class =17: No entry                                              |120 |  120(100%)  |   0(0  %)   |      8  |
|class =18: General caution                                       |120 |  117(97 %)  |   3(2  %)   |      2  |
|class =19: Dangerous curve to the left                           | 30 |   30(100%)  |   0(0  %)   |      0  |
|class =20: Dangerous curve to the right                          | 60 |   50(83 %)  |  10(16 %)   |      1  |
|class =21: Double curve                                          | 60 |   41(68 %)  |  19(31 %)   |      0  |
|class =22: Bumpy road                                            | 60 |   58(96 %)  |   2(3  %)   |      2  |
|class =23: Slippery road                                         | 60 |   58(96 %)  |   2(3  %)   |      1  |
|class =24: Road narrows on the right                             | 30 |   29(96 %)  |   1(3  %)   |      3  |
|class =25: Road work                                             |150 |  147(98 %)  |   3(2  %)   |      1  |
|class =26: Traffic signals                                       | 60 |   59(98 %)  |   1(1  %)   |      3  |
|class =27: Pedestrians                                           | 30 |   26(86 %)  |   4(13 %)   |      0  |
|class =28: Children crossing                                     | 60 |   60(100%)  |   0(0  %)   |      3  |
|class =29: Bicycles crossing                                     | 30 |   30(100%)  |   0(0  %)   |      1  |
|class =30: Beware of ice/snow                                    | 60 |   60(100%)  |   0(0  %)   |      2  |
|class =31: Wild animals crossing                                 | 90 |   90(100%)  |   0(0  %)   |     16  |
|class =32: End of all speed and passing limits                   | 30 |   30(100%)  |   0(0  %)   |     12  |
|class =33: Turn right ahead                                      | 90 |   90(100%)  |   0(0  %)   |      1  |
|class =34: Turn left ahead                                       | 60 |   60(100%)  |   0(0  %)   |      0  |
|class =35: Ahead only                                            |120 |  120(100%)  |   0(0  %)   |      1  |
|class =36: Go straight or right                                  | 60 |   60(100%)  |   0(0  %)   |      0  |
|class =37: Go straight or left                                   | 30 |   30(100%)  |   0(0  %)   |      0  |
|class =38: Keep right                                            |210 |  210(100%)  |   0(0  %)   |      0  |
|class =39: Keep left                                             | 30 |   30(100%)  |   0(0  %)   |      0  |
|class =40: Roundabout mandatory                                  | 60 |   59(98 %)  |   1(1  %)   |      0  |
|class =41: End of no passing                                     | 30 |   22(73 %)  |   8(26 %)   |     23  |
|class =42: End of no passing by vehicles over 3.5 metric tons    |  30|    30(100%) |    0(0  %)  |       0 |
| <u>TOTAL</u>                                                    | <u>4410</u>  |      <u>4270</u>  |       <u>140</u>    |   <u>140</u>    |


xxIokPyiO$326
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web, shown in RGB, GrayScale and CLAHE:

![New Signs from the Web](https://github.com/ArjaanBuijk/CarND_Traffic_Sign_Classifier_Project/blob/master/signs_new.jpg)

I made sure to include a speed limit sign and some danger signs. Those are the classes that were the hardest to predict during testing phase.

In general, I expect that the classifier will be able to give a good prediction on each of these. They are clear, good contrast, and not obstructed by other objects. 
  
#### 2. Predictions of new images from the web.

The code for loading the images and pre-process them is in cell 23 of the Jupyter notebook.
The code for predicting the labels with the trained predictor is in cell 24of the Jupyter notebook.

Because I store all 6 images and labels in the numpy arrays X_new and y_new, the coding is identical to what was used to predict the test set.

When I tested these new images on a predictor that was trained with non-augmented data, it often had the 4th image wrong (class 18, General caution). Once I switched to the augmented data set, it always got 100 % of these tests correct.

 The accuracy of the new images is 100% while it was 95% on the testing set. This shows that the classifier is well able to correctly predict unseen images and is not over-fitted to the training/validation/test images.

####3. Softmax probabilities for new images from web

The code for determining the top 5 softmax probabilities for each new image from the web is located in the 25th and 26th cell of the Jupyter notebook.

For all images, the model is very sure about the prediction.

<u>Top 5 softmax probabilities for each of the new test images</u>

![prediction_new_class2](https://github.com/ArjaanBuijk/CarND_Traffic_Sign_Classifier_Project/blob/master/prediction_new_class2.jpg)

---

![prediction_new_class11](https://github.com/ArjaanBuijk/CarND_Traffic_Sign_Classifier_Project/blob/master/prediction_new_class11.jpg)

---

![prediction_new_class14](https://github.com/ArjaanBuijk/CarND_Traffic_Sign_Classifier_Project/blob/master/prediction_new_class14.jpg)

---

![prediction_new_class18](https://github.com/ArjaanBuijk/CarND_Traffic_Sign_Classifier_Project/blob/master/prediction_new_class18.jpg)

---

![prediction_new_class25](https://github.com/ArjaanBuijk/CarND_Traffic_Sign_Classifier_Project/blob/master/prediction_new_class25.jpg)

---

![prediction_new_class31](https://github.com/ArjaanBuijk/CarND_Traffic_Sign_Classifier_Project/blob/master/prediction_new_class31.jpg)
