# PNEUMONIA PREDICTING DEEP LEARNING PROJECT
MICROSOFT STUDENT ACCELERATOR 2020 - AI & ADVANCED ANALYTICS

- Written by Dat Huynh
- Email: viendat.huynh@student.unsw.edu.au
- School of Computer Science and Engineering, University of New South Wales

## 1. PROJECT IDEA
This project develops a Convolutional Neural Network model for pneumonia prediction. The model is fit with chest X-ray images of both pneumonia patients and normal people. Then it predicts if a person gets pneumonia through analysing his/her chest X-ray image.

### Practicality
As machine learning has been widely applied in medicine, it's potentials and future applications in this field are indisputable. From beginning of 2020, the world has faced the Corona pandemic and probably more diseases in the future. An effective prediction model can assist doctors in disease diagnosis and contain the pandemics. Initially, this project was developed for predicting COVID-19 but then shifted to a simpler task (pneumonia prediction) due to the lack of COVID-19 data.

### Uniqueness & Novelty
There are already several projects on this topic such as:
- Researchers develop a new system that can distinguish pneumonia from COVID-10 in chest X-rays
https://medicalxpress.com/news/2020-05-distinguish-pneumonia-covid-chest-x-rays.html
- An Efficient Deep Learning Approach to Pneumonia Classification in Healthcare
https://www.hindawi.com/journals/jhe/2019/4180949/

This proves the applicability of pneumonia prediction. All projects with large amount of data and great resources have been able to achieve the accuracy of over 90%. The main goal of this project is to explore Deep Learning and Convolutional Neural Networks model development so an accuracy rate of over 75% is acceptable.

### Future Scalability
Since this CNN is able to classify pneumonia and normal chest X-ray by analysing X-ray images, it is possible to develop the model for classifying pneumonia causes like stress-smoking, virus or bacteria. We can even go further to make classification of viruses that cause pneumonia like COVID-19, SARS or Streptococcus if a larger amount of higher quality data is fit.

## 2. ENVIRONMENT SETUP
### Dataset
This project uses train and test dataset from CoronaHack Chest X-Ray Dataset:
https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset/data?select=Coronahack-Chest-XRay-Dataset 

It contains about 6000 train and 600 test chest X-ray images of both pneumonia patients and normal people. At the first glance, this dataset seems to be suitable for COVID-19 prediction. In fact, there are only 58 COVID-19, 4 SARS and 2 ARDS samples which are very deficient. With this set, it is just possible to build an effective model for normal and pneumonia prediction. And maybe a pneumonia cause classification model (virus or bacteria) could be developed if we spend a bit more time to process the dataset.

### Machine Learning Model
This is a classification and computer vision problem so deep learning or more specifically, Covolutional Neural Networks are the most commonly used. Using TensorFlow 2.2.0, the model converts JPEG and JPG files into pairs of 2D grayscale images and HotVector labels. Then it is trained with datasets of different image resolutions and sizes. Also, it is tested with several activation functions and optimizer to achieve the best performance.

## 3. IMPLEMENTATION
### Dataset Directory
Create TensorFlow datasets that contains all the directory paths of image files in the folder 'dataset/Coronahack-Chest-XRay-Dataset'.
- Import the CSV file with image names and labels from 'dataset/Chest_xray_Corona_Metadata.csv'
- As the number of COVID-19, ARDS, SARS and Streptococcus samples is very deficient, the model uses only the attribute 'X_ray_image_name' and 'Label'. We also convert 'Label' to binary values ("Normal": 0, "Pnemonia": 1) for optimising the performance.
- Create a list of all paths to images in train and test folders. Preview them to make sure the dataset works.

### TensorFlow Directory Dataset
Before processing images and labels, two TensorFlow datasets (training and testing) are created containing JPEG file paths.
- Create 2 TensorFlow datasets of the file paths to 2 folders 'dataset/Coronahack-Chest-XRay-Dataset/train' and 'dataset/Coronahack-Chest-XRay-Dataset/test'.

### Dataset Quality Setting
> **NOTE:** Using image resolutions higher than 280x200 will result in significant long training time or even kernel's death if the processor is not enough powerful. After several tests, the resolution 140x100 is the most balance between performance and accuracy. In fact, it even achieves a higher accuracy with test dataset than better resolutions like 280x200 and 420x300. The notebook set default as below for a quick overview. To get better performance, set IMG_HEIGHT = 100, IMG_WIDTH = 140 and increase the BATCH_TRAIN_SIZE to 2000-5000 and BATCH_TEST_SIZE to 200-600.

### Process Train & Test Dataset
Convert the TensorFlow datasets to datasets of grayscale Tensor images and labels.
- Use Dataset.map to create a dataset of Tensor (image, label) pairs.
- Batch the images in train and test sets then create a validation set from train set.
- Convert images to grayscale (1 dimension) and Numpy array to fit the model.
- Convert labels to One Hot vectors 0: [1. 0.] (Normal) and 1: [0. 1.] (Pneumonia)

### Build Convolutional Neural Network
- Use the model Sequential() and setup pre-processing layer of the network
- Add pooling layers MaxPooling2D to speed up training time and make features it detects more robust
- Add Dropout layer to prevent overfitting
- Add Dense layers to perform classification
- Compile with an optimizer

### Train & Evaluate The Model
- Fit the model with train and validation datasets
- Plot the training statistics to see how it's performance developed over time

## 4. OUTCOME
### Accuracy
Some evaluation results when training the model with dataset of different sizes, quality, activations and optimizers:
- Size: 2000, Image Resolution: 50 x 70
activation = 'softsign', epochs = 10
  loss = 0.94, accuracy = 75.50
 
- Size: 2000, Image Resolution: 100 x 140
activation = 'softsign', epochs = 10
  loss = 0.93, accuracy = 87.00
  
- Size: 3000, Image Resolution: 100 x 140
activation = 'softsign', epochs = 10
  loss = 0.93, accuracy = 77.20
  
- Size: 5000, Image Resolution: 50 x 70
activation = 'softsign', epochs = 10
  loss = 0.95, accuracy = 78.80

- Size: 5000, Image Resolution: 100 x 140
activation = 'softsign', epochs = 12
  loss = 0.96, accuracy = 76.80

### Analysis
- Testing activation functions ['elu', 'selu', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'softplus', 'softsign', 'linear'] shows that 'softsign' and 'linear' perform the best. However, 'softsign' is more consistent and accurate when the sizes of train and test sets increase. 
- Also, Among optimizers ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'], 'Adam' gives the most consistent and accurate performance.
Noticeably, the network achieves the best performance with dataset of 100x140 images which have lower resolution than 200x280 or 300x420. Moreover, the model fit with only 2000 samples reaches higher accuracy rate at 87% with 200 test samples and 83% with 600 test samples. This is even more accurate than the model fit with full train set (6000 samples) tested by 600 test samples. It is clear that a deep neural network sometimes performs better with less data and lower resolution train images.

## 5. CONCLUSION
Throughout this project, I have aquired significant amount of knowledge about Deep Learning and Neural Network. I realised how important the dataset and pre-processing is to model training. I have also been able to develop Python Notebook skills in data processing and CNN model building from scratch. However, this model has been only trained and tested on average hardware so it is not possible to try the better resolution images and higher epochs to see if the performance can improve. I would like to try this on a more powerful processor in the future.
