#**Traffic Sign Recognition**
**Build a Traffic Sign Recognition Project**

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

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

Here is a link to my [project code (Jupyter Notebook)](https://github.com/ArjaanBuijk/CarND_Traffic_Sign_Classifier_Project/blob/master/Traffic_Sign_Classifier.ipynb)

Here is a link to my [project code (HTML exported after run)](https://github.com/ArjaanBuijk/CarND_Traffic_Sign_Classifier_Project/blob/master/Traffic_Sign_Classifier.html)
  
 NOTE: The github website will not display the HTML file, because it is too big. You must download it to your disk, and then open it in your browser to view it.

###Imports

####All import statements are placed at the top of the notebook.

The code for this step is contained in the first code cell of the Jupyter notebook. 


###Hyper Parameters

####All hyper parameters are defined as global variables, and defined in a single code cell, to allow easy optimization studies.

The code for this step is contained in the second code cell of the Jupyter notebook. 

The following hyper parameters are available for optimization:

| Hyper Parameter | Description | Optimized Value |
| --------------- | ----------- | --------------- |
| EPOCHS | how many forward/backward loops | 200 |
| BATCH_SIZE| training images per batch | 16 |
| rate| learning rate | 0.0001 |
| CLIP_LIMIT | clip limit during CLAHE (see below) | 0.1 |
 
I ran many variations of these 4 parameters, and ended up with the optimized values given in above table.  Most surprising to me was the fact that a small value for the BATCH_SIZE gave the highest accuracy. 

###Loading data

####All data was downloaded, stored in a folder and loaded from disk into numpy arrays.

The code for this step is contained in the third code cell of the Jupyter notebook. 

The downloaded pickle files are stored in the folder: ./data
It was already sub-divided into a training, validation and test data set.

The images (X) and the labels (y) are read into these numpy arrays:

 - X_train , y_train
 - X_valid, y _valid
 - X_test, y_test
 
These are the training, validation and test sets respectively.
  

###Data Set Summary & Exploration

####1. The data sets were first investigated for some basic information.

The code for this step is contained in the fourth code cell of the Jupyter notebook.  

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

 

####2. More in depth, visual investigation of the data.

The code for this step is contained in the fifth code cell of the Jupyter notebook.  

First, the description belonging to each class is read into a Panda DataFrame, from the file signnames.csv.

This allows for clear descriptions in the reports.

Then, in this code cell are two functions that were used during this initial investigation, and then re-used below during actual pre-processing of the data.

| Function | Description |
| -------- | ----------- |
|grayscale | converts an RGB image into grayscale | 
|apply_clahe | Applies a Contrast Limited Adaptive Histogram Equalization (CLAHE)|

The CLAHE technique was found during internet research on image processing to get more contrast in the pictures, and the method used is from the [scikit.exposure](http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist) library.

During training, it was found that applying the CLAHE technique was very beneficial to get higher accuracy.

The 3rd function in this code cell creates a table that summarizes for each class:

 - The class label
 - The description of the class
 - The number of images in the training set for this class
 - The RGB image of the first item in the training set 
 - The image after Grayscale
 - The image after Grayscale and CLAHE
 
![Table of Training Images](https://github.com/ArjaanBuijk/CarND_Traffic_Sign_Classifier_Project/blob/master/signs_training_summary.jpg)

###Design and Test a Model Architecture

####1. The training data is pre-processed using augmentation, Grayscale conversion and Contrast Limited Adaptive Histogram Equalization (CLAHE)

The code for this step is contained in the fifth to eight code cells of the Jupyter notebook.

During detailed investigation of the images, as described above, I found 3 issues with the training set:

 1. There was a large difference in number of images per class.
 2. The traffic signs do NOT require color to be taken into account for classification.
 3. Even after conversion to Grayscale, there were a lot of images with very poor contrast.

These issues were addressed by the following 3 pre-processing techniques:

 1. I augmented the training data set, and added images to each class by taking the existing images and applying a random rotation. After this step, each class ended up with the same number of images. The total number of images increased from 34799 to 86430.
 2. I converted all training, validation and test images to Grayscale.
 3. After that, I applied CLAHE, which is both an equalizer and a normalization.

The effect of Grayscale and CLAHE was already described and shown in the previous section. 

At the end of the pre-processing steps, the data is written to a pickle file. Especially the augmentation and CLAHE are time consuming, and it is important to have to redo this all the time during hyper-parameter optimization.

####2. The model architecture and utility functions.

The code for my final model is located in the ninth to thirteenth cells of the Jupyter notebook. 

<u>Description of LeNet-5 function:</u>

I made the following modifications to the LeNet-5 implementation:

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

<u>Description of utility functions:</u>

| function | Description |
| -------- | ----------- |
|evaluate|Evaluate the accuracy of predictions|
|prediction_counts|Count number of true-positive, false-negative, false-positive predictions for each class|
|summarize_predictions|Summarizes predictions in tabular and visual formats|
|summarize_top_probabilities|Summarizes top probabilities in tabular format|
 
 
####4. Training approach.

The code for loading the pre-processed data is contained in the fourteenth cell of the Jupyter notebook. 
The training pipeline and training session is defined in the tenth and fifteenth cell of the Jupyter notebook.

To train the model, I used the AdamOptimizer.

I simply ran many variations of the hyper-parameters described above, in a systematic manner. Varying one parameter at a time. Each time, I looked at the validation accuracy overall, but also at the prediction Summary, like this:

<u>Prediction summary:</u>
 
| label | count | true-pos | false-neg | false-pos |
| ----- | ----- | -------- | --------- | --------- |
|class =0: Speed limit (20km/h)                                 |   30   | 23(76 %)   |  7(23 %) |        1  |
|class =1: Speed limit (30km/h)                                 |  240   |233(97 %)   |  7(2  %) |       12  |
|class =2: Speed limit (50km/h)                                 |  240   |235(97 %)   |  5(2  %) |        5  |
|class =3: Speed limit (60km/h)                                 |  150   |141(94 %)   |  9(6  %) |        4  |
|class =4: Speed limit (70km/h)                                 |  210   |208(99 %)   |  2(0  %) |        6  |
|class =5: Speed limit (80km/h)                                 |  210   |199(94 %)   | 11(5  %) |        9  |
|class =6: End of speed limit (80km/h)                          |   60   | 59(98 %)   |  1(1  %) |        1  |
|class =7: Speed limit (100km/h)                                |  150   |150(100%)   |  0(0  %) |        1  |
|class =8: Speed limit (120km/h)                                |  150   |149(99 %)   |  1(0  %) |        4  |
|class =9: No passing                                           |  150   |148(98 %)   |  2(1  %) |        0  |
|class =10: No passing for vehicles over 3.5 metric tons        |  210   |210(100%)   |  0(0  %) |        0  |
|class =11: Right-of-way at the next intersection               |  150   |150(100%)   |  0(0  %) |        7  |
|class =12: Priority road                                       |  210   |210(100%)   |  0(0  %) |        3  |
|class =13: Yield                                               |  240   |237(98 %)   |  3(1  %) |        2  |
|class =14: Stop                                                |   90   | 83(92 %)   |  7(7  %) |        0  |
|class =15: No vehicles                                         |   90   | 90(100%)   |  0(0  %) |        0  |
|class =16: Vehicles over 3.5 metric tons prohibited            |   60   | 30(50 %)   | 30(50 %) |        0  |
|class =17: No entry                                            |  120   |114(95 %)   |  6(5  %) |        0  |
|class =18: General caution                                     |  120   |118(98 %)   |  2(1  %) |       13  |
|class =19: Dangerous curve to the left                         |   30   | 30(100%)   |  0(0  %) |        5  |
|class =20: Dangerous curve to the right                        |   60   | 35(58 %)   | 25(41 %) |        4  |
|class =21: Double curve                                        |   60   | 32(53 %)   | 28(46 %) |        0  |
|class =22: Bumpy road                                          |   60   | 59(98 %)   |  1(1  %) |        1  |
|class =23: Slippery road                                       |   60   | 57(95 %)   |  3(5  %) |       14  |
|class =24: Road narrows on the right                           |   30   | 23(76 %)   |  7(23 %) |        4  |
|class =25: Road work                                           |  150   |142(94 %)   |  8(5  %) |        3  |
|class =26: Traffic signals                                     |   60   | 54(90 %)   |  6(10 %) |        2  |
|class =27: Pedestrians                                         |   30   | 24(80 %)   |  6(20 %) |        8  |
|class =28: Children crossing                                   |   60   | 60(100%)   |  0(0  %) |        7  |
|class =29: Bicycles crossing                                   |   30   | 30(100%)   |  0(0  %) |        2  |
|class =30: Beware of ice/snow                                  |   60   | 58(96 %)   |  2(3  %) |        4  |
|class =31: Wild animals crossing                               |   90   | 90(100%)   |  0(0  %) |       12  |
|class =32: End of all speed and passing limits                 |   30   | 30(100%)   |  0(0  %) |       32  |
|class =33: Turn right ahead                                    |   90   | 87(96 %)   |  3(3  %) |        3  |
|class =34: Turn left ahead                                     |   60   | 59(98 %)   |  1(1  %) |        1  |
|class =35: Ahead only                                          |  120   |120(100%)   |  0(0  %) |        1  |
|class =36: Go straight or right                                |   60   | 60(100%)   |  0(0  %) |        4  |
|class =37: Go straight or left                                 |   30   | 30(100%)   |  0(0  %) |        1  |
|class =38: Keep right                                          |  210   |199(94 %)   | 11(5  %) |        1  |
|class =39: Keep left                                           |   30   | 30(100%)   |  0(0  %) |       15  |
|class =40: Roundabout mandatory                                |   60   | 58(96 %)   |  2(3  %) |        1  |
|class =41: End of no passing                                   |   30   | 27(90 %)   |  3(10 %) |        1  |
|class =42: End of no passing by vehicles over 3.5 metric tons  |    30  |  30(100%)  |   0(0  %)|         5 |
|<u>TOTAL</u>                               | <u>4410</u>   |     <u>4211</u>   |      <u>199</u> |      <u>199</u>  |

When using a non-augmented data set I could actually achieve a higher overall accuracy, but there were many classes with a 30+ % of false negatives. With the augmented data set, the overall accuracy was lower, but there were not many classes with such a high rate of false negatives.

The only ones remaining were class 16, 20 and 21. 

In addition to the tabular format, I also plotted 2 of the false negatives for each class, to get an idea why it missed the prediction. From the images plotted, it was clear that:

 - The predictor is good at detecting a speed limit sign, but sometimes missed the speed limit value. For example, it predicted 70 km/h instead 20 km/h.
 - Several 'danger'signs, warning for traffic signals, curve to right, curve to left, double curve, etc.. were mis-predicted.


My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.955 
* test set accuracy of 0.932

 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web, shown in RGB, GrayScale and CLAHE:

![New Signs from the Web](https://github.com/ArjaanBuijk/CarND_Traffic_Sign_Classifier_Project/blob/master/signs_new.jpg)

####2. Predictions of new images from the web.

The code for loading the images and pre-process them is in cell 18 of the Jupyter notebook.
The code for predicting the labels with the trained predictor is in cell 19 of the Jupyter notebook.

Becuase I store all 5 images and labels in the numpy arrays X_new and y_new, the coding is identical to what was used to predict the test set.

When I tested these new images on a predictor that was trained with non-augmented data, it often had the 3rd image wrong. Once I switched to the augmented data set, it always got 100 % of these tests correct.

I realize that I should not determine hyper-parameters based on the test set, and would have to find more images to make make sure my predictor is well trained. I will leave this for later at the moment.

I did not study the effect of modifying the architecture of the network, except the addition of the dropout layer already described above. 

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.2%

####3. Softmax probabilities for new images from web

The code for determining the top 5 softmax probabilities for each new image from the web is located in the 21st cell of the Jupyter notebook.

For all images, the model is very sure about the prediction, as can be seen from this table:

<u>Top 5 softmax probabilities</u>

class =11: Right-of-way at the next intersection
Predictions   :['        11', '        30', '        27', '        40', '        25']
Probabilities :['   <b>100.00%'</b>, '     0.00%', '     0.00%', '     0.00%', '     0.00%']

class =14: Stop
Predictions   :['        14', '        15', '         3', '         8', '        35']
Probabilities :['    <b>76.65%</b>', '    22.99%', '     0.27%', '     0.08%', '     0.01%']

class =18: General caution
Predictions   :['        18', '        27', '        26', '         0', '         4']
Probabilities :['   <b>100.00%</b>', '     0.00%', '     0.00%', '     0.00%', '     0.00%']

class =25: Road work
Predictions   :['        25', '        37', '        18', '        20', '        34']
Probabilities :['   <b>100.00%</b>', '     0.00%', '     0.00%', '     0.00%', '     0.00%']

class =31: Wild animals crossing
Predictions   :['        31', '        21', '        23', '        19', '        25']
Probabilities :['   <b>100.00%</b>', '     0.00%', '     0.00%', '     0.00%', '     0.00%']