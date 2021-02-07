# SDCNC_Behavior_Cloning


# Behaviorial Cloning Project

The steps of this project are the following:

Use the simulator to collect data of good driving behavior      

Build, a convolution neural network in Keras that predicts steering angles from images

Train and validate the model with a training and validation set

Test that the model successfully drives around track one without leaving the road

Summarize the results with a written report

### According to Rubric points, below is the report for the project 

### Files Submitted & Code Quality

1) Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

model.py containing the script to create and train the model

drive.py for driving the car in autonomous mode

model.h5 containing a trained convolution neural network

writeup_report.md or writeup_report.pdf summarizing the results

2)  Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

python drive.py model.h5


3)  Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used

for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1) An appropriate model architecture has been employed

Initially the LeNet model was used but there were some issue in the final simulation as the model was unable to generate the required steering output. The car always swayed away from the road. Although more data was fed, I felt the need to move to more powerful model. Hence I used the Nvidia Model. This increased the number of parameters therefore, data augmentation was reduced for reduction in computational time.

My model consists of a convolution neural network with multiple filter sizes and depths between 24 and 64.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer .

The model had: 

-5 conv layers. First3 layers with 5*5 kernel size and strides. Last 2 conv layers had 3*3 kernel size and strides.

-a Flatten layer

-3 FC (Dense) layers with 100,50,1 neurons each. 

Since this was regression and not classification, no activation for the last layer was used. Also, the cost funtion was 'mean squared error' rather than 'cross entropy'.

#### 2) Attempts to reduce overfitting in the model

The LeNet model contained dropout layers in order to reduce overfitting. But the with the nvidia model, less overfiting was observed and use of dropout layers was not required. the number of epochs was also reduced to over training the model.

The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3) Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. 'RMS prop was also tried and results were a similar and only slower. Hence, I moved back to Adam optimizer. Number of epochs were optimized by avoiding overfitting. I played with the kernel and strides sizes, and the current ones were already optimzied.

#### 4) Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. An augmented data (flipped) was also appended into the current dataset. For data:

I took 2 whole track data with smooth driving . 1 track of recovery around sharp corners. I also added the data from track 2 to generalize the model. But it made the model worse, so I had to removed it. Anyways, the model somehow covers track safely in autonomous mode.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1) Solution Design Approach

The overall strategy for deriving a model architecture was to simulate the track 1 in autonomous mode which I was succesful in.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it was a powerful network and well known for image classification/regression. Though, it gave good improvements, it was not quite ready for submission. I used nVidia model as advised by the Udacity team.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set and a little higher mean squared error on the validation set. This implied that the model was overfitting, but not a lot.

To combat the overfitting, I modified the model so that added dropout layers after each convolutional layers. This only slighlty reduced the validation loss, nonetheless, dropout layers were removed and the model was running good on the simulator on the autonomous mode.

Then I used 3 fully connected layers. All the layers had Relu to incorporate non-linearity except the last layer because this problem is a regression problem and nopt a classification problem.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and failed to revocer intially. To improve the driving behavior in these cases, I added more recovery data in these areas. My model specifically struggled in the 'no curb' patch, but after a good amount of data in these areas, it ran fine.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2) Final Model Architecture

My model consists of a convolution neural network with multiple filter sizes and depths between 24 and 64.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer .

The model had:

-5 conv layers. First3 layers with 55 kernel size and strides. Last 2 conv layers had 33 kernel size and strides.

-a Flatten layer

-3 FC (Dense) layers with 100,50,1 neurons each.

Since this was regression and not classification, no activation for the last layer was used. Also, the cost funtion was 'mean squared error' rather than 'cross entropy'.

![Model Architecture](Arch.PNG)

The difference is in the input image size. The input iamge size used by me is 160,320,3.

#### 3) Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:
![Center](center.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the center of the road if it drifts to sides. 
![left](left.jpg) ![center](center.jpg) ![right](right.jpg)

To augment the data set, I also flipped images thinking that this would diversify the dataset and would make a more generalized model instead of model designed only for track1. For example, here is an image that has then been flipped. 
![Flipped](center_flipped.jpg)

After the collection process, I had 37464 number of data points. I then preprocessed this data by normalizing this dataset using lambda layer with an input size of 160,320,3.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 which was monitored by watch the validation loss. If it increases, the number of epochs were reduced. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### P.S.

I tried using generator but was facing a memory leak somehow. I was unable to find the leak. The error was : Memory error. Therefore, I trained the conventional way, altough I have attached the generator code in the model.py as well.


```python

```


```python

```
