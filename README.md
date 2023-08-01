# Skin Cancer Detection: HAM10000

## Skin cancer overview 

Skin cancer is a type of cancer that develops in the skin due to the uncontrolled growth of abnormal cells, often caused by exposure to ultraviolet (UV) radiation from the sun or artificial sources. The three main types of skin cancer are basal cell carcinoma, squamous cell carcinoma, and melanoma, with melanoma being the most aggressive and potentially life-threatening. Early detection and sun protection are essential in preventing and managing this disease. Skin cancer can appear in various ways, depending on the type and stage. Basal cell carcinoma often presents as a pearly or waxy bump, a pinkish patch, or a sore that doesn't heal. Squamous cell carcinoma may manifest as a scaly, red patch, an ulcerated sore, or a raised growth with a central depression. Melanoma typically shows up as a new, unusual-looking mole or an existing mole that undergoes changes in size, shape, color, or texture. It's crucial to be vigilant for any suspicious skin changes and seek prompt evaluation by a dermatologist if you notice anything concerning, as early detection significantly improves the chances of successful treatment. The primary cause of skin cancers is exposure to ultraviolet (UV) radiation from the sun or artificial sources like tanning beds. UV radiation damages the DNA in skin cells, leading to mutations that can trigger uncontrolled cell growth and the development of cancerous cells. Other risk factors include having fair skin, a history of sunburns, a weakened immune system, a family history of skin cancer, and exposure to certain chemicals or radiation. Practicing sun protection and avoiding artificial UV sources are essential in reducing the risk of developing skin cancer.

Skin cancer dangers include:
* Metastasis
* Disfigurement and Tissue Damage
* Difficult Treatment
* Recurrence
* Secondary Skin Cancers

## Business Problem



## Why this project? 
According to one estimate, approximately 9,500 people in the U.S. are diagnosed with skin cancer every day. Skin cancer is also the most common cancer in the United States and is appearing more and more in populations with excess exposure to UV radiation. A image classifiaction model to detect certain types of skin diseases is useful in diagnosing and preparing to treate such a disease since it is eaily accessable woth the deployed application. Indiviuails without readily access to healthcare who are hesitant to seek medical help can use this projet to test themselves and have self detemination to decide of they are at risk. While it does not replace the need to see a doctor or dermatilogist, it is a step in raing awareness to those who belive that a harmelss freckle or mole is a non threatning thing then their lives may be at risk. 



## About the data

The data originates from two distinct locations. One being the HAM10000 dataset from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
The HAM10000 dataset from Harvard Dataverse, aims to address the challenges in training neural networks for automated diagnosis of pigmented skin lesions. The dataset contains 10015 dermatoscopic images from diverse populations, acquired and stored using different modalities. It serves as a valuable training set for academic machine learning purposes. The dataset covers various diagnostic categories of pigmented lesions, including actinic keratoses, basal cell carcinoma, benign keratosis-like lesions, dermatofibroma, melanoma, melanocytic nevi, and vascular lesions.More than 50% of the lesions in the dataset have been confirmed through histopathology, while the ground truth for the remaining cases is determined through follow-up examination, expert consensus, or confirmation by in-vivo confocal microscopy. Additionally, the dataset includes lesions with multiple images, allowing them to be tracked by the lesion_id-column within the HAM10000_metadata file.

The other source is [Kaggle, Dermatology MNIST](https://www.kaggle.com/code/kmader/dermatology-mnist-loading-and-processing). The data taken from kaggle is linked in the dermatology-mnist-loading-and-processing.ipynb file which proceses the images into pixel arrays(data) used in the main model notebook. The code flattening the images into vectors and exports them. I ran this notebook as is and did not edit any of this notebook. 

## Resutls:
My CNN built model preformed better than the tranfer learning models. For example the CNN accuracy reached 76% with a auc score of 96%. The MobilenetV2 model which was going to be used in deployment only reached an accuracy of 71% and a auc score of 94%. This could be due to poor engineering of the MobilenetV2 on my part and in futur work i will imporve on this model. 

## Defining my Convolutional Neural Network
The function blockred defines a custom block architecture for a neural network in Python using the Keras/TensorFlow library. The block consists of three branches: Inception, VGG + SqueezeNet, and InceptionResNet + Squeeze. Each branch performs a series of convolutional operations on the input inp, which is a 2D convolutional tensor, with a specified number of filters. Within each branch, convolutional layers with different filter sizes, pooling layers, and batch normalization are used to extract and process features from the input tensor. Finally, the outputs of the three branches are element-wise added together, and the result is returned as output1. This block can be used as a building block within a larger neural network to create more complex architectures for various computer vision tasks, such as image classification, object detection, or segmentation.

The model named "CNN0" is a Convolutional Neural Network (CNN) designed for image classification tasks. It takes input images with a shape of (28, 28, 3), representing 28x28 RGB images. The model consists of several convolutional blocks, each with convolutional layers, batch normalization, and dropout for regularization. The output of the last block is flattened and passed through fully connected layers with ReLU activation and dropout. The final layer is a fully connected layer with 7 units and a softmax activation function, representing the model's output probabilities for a 7-class classification task. The model's architecture, including the layer types, output shapes, and trainable parameters, is summarized and printed.

<img width="487" alt="Screenshot 2023-08-01 at 12 32 57 AM" src="https://github.com/jj862/Skin-detection-CNN/assets/69119958/ca596a98-3d7a-4617-bcf8-0c00e5e3ec67">
<img width="467" alt="Screenshot 2023-08-01 at 12 34 27 AM" src="https://github.com/jj862/Skin-detection-CNN/assets/69119958/a5a74ae3-e9ad-4e23-8dc3-a7ac394d1c36">
<img width="659" alt="Screenshot 2023-08-01 at 12 35 05 AM" src="https://github.com/jj862/Skin-detection-CNN/assets/69119958/78a4791e-9c20-4fa4-8ca2-98175ba06a43">
<img width="384" alt="Screenshot 2023-08-01 at 12 35 42 AM" src="https://github.com/jj862/Skin-detection-CNN/assets/69119958/5e3b8df4-8154-4b7c-b728-5057c89c922d">


## MobilenetV2
MobileNetV2 is a specific neural network architecture designed to address the challenges of deploying deep learning models on resource-constrained devices, such as mobile phones, embedded systems, and IoT devices. The main reasons for using MobileNetV2 are low memory footprint and mobile Applications. Given the rise of mobile applications and the need for on-device AI capabilities, MobileNetV2 is a popular choice for developers looking to deploy deep learning models on mobile devices, where resource constraints are significant considerations. However for this notebook, "CNN0" will be used in development due to better results. 
<img width="661" alt="Screenshot 2023-08-01 at 12 46 23 AM" src="https://github.com/jj862/Skin-detection-CNN/assets/69119958/1d8ab91c-4802-46f3-b503-d2d36f907541">
<img width="377" alt="Screenshot 2023-08-01 at 12 46 53 AM" src="https://github.com/jj862/Skin-detection-CNN/assets/69119958/91c43895-6b71-460a-bcb9-69bbbe293172">


## Difficutultes and future reccomendations 
While there are over 10000 images to train, there is a huge imbalance in the data. For instance, melanocytic nevi appears about 6705 times while Dermatofibroma only appears 115 times. While image augmentation and standardization is used to fix this imbalance, I believe it drops the accuracy of the model dramatically. Scores when standatdiztion was done before the test train split were done reached around 90% and higher. But this creates overfitting. When done properly, an accuracy of 76% was achivable usng my CNN model. For future work i will include external data to address this issue. 
<img width="493" alt="Screenshot 2023-08-01 at 12 42 10 AM" src="https://github.com/jj862/Skin-detection-CNN/assets/69119958/d0152601-a91b-4245-9989-403b3ded51c0">


Another issue was the amount of classes used in the model. The data was tested on 7 different classes, and for future work I will address each of these classes seperatly rather than all at once to improve accuracy.

## Contact
jonah.devoy@yahoo.com
www.linkedin.com/in/jonahdevoy

## Glossary:
- Precision - Indicates the proportion of positive identifications (model predicted class `1`) which were actually correct. A model which produces no false positives has a precision of 1.0.
- Recall - Indicates the proportion of actual positives which were correctly classified. A model which produces no false negatives has a recall of 1.0.
- F1 score - A combination of precision and recall. A perfect model achieves an F1 score of 1.0.
- Support - The number of samples each metric was calculated on.
- Accuracy - The accuracy of the model in decimal form. Perfect accuracy is equal to 1.0, in other words, getting the prediction right 100% of the time.
- Macro avg - Short for macro average, the average precision, recall, and F1 score between classes. The Macro avg doesn't take class imbalance into effect. So if you do have class imbalances (more examples of one class than another), you should pay attention to this.
- Weighted avg - Short for the weighted average, the weighted average precision, recall, and F1 score between classes. Weighted means each metric is calculated with respect to how many samples there are in each class. This metric will favor the majority class (e.g. it will give a high value when one class outperforms another due to having more samples).

- False negatives — Model predicts negative, actually positive. In some cases, like email spam prediction, false negatives aren’t too much to worry about. But if a self-driving car's computer vision system predicts no pedestrian when there was one, this is not good.
- False positives — Model predicts positive, actually negative. Predicting someone has heart disease when they don’t, might seem okay. Better to be safe right? Not if it negatively affects the person’s lifestyle or sets them on a treatment plan they don’t need.
- True negatives — Model predicts negative, actually negative. This is good.
- True positives — Model predicts positive, actually positive. This is good.
- Precision — What proportion of positive predictions were actually correct? A model that produces no false positives has a precision of 1.0.
- Recall — What proportion of actual positives were predicted correctly? A model that produces no false negatives has a recall of 1.0.
- F1 score — A combination of precision and recall. The closer to 1.0, the better.
- Receiver operating characteristic (ROC) curve & Area under the curve (AUC) — The ROC curve is a plot comparing true positive and false positive rate. The AUC metric is the area under the ROC curve. A model whose predictions are 100% wrong has an AUC of 0.0, and one whose predictions are 100% right has an AUC of 1.0.

- R^2 (pronounced r-squared) or the coefficient of determination** - Compares your model's predictions to the mean of the targets. Values can range from negative infinity (a very poor model) to 1. For example, if all your model does is predict the mean of the targets, its R^2 value would be 0. And if your model perfectly predicts a range of numbers its R^2 value would be 1. 
- Mean absolute error (MAE) - The average of the absolute differences between predictions and actual values. It gives you an idea of how wrong your predictions were.
- Mean squared error (MSE) - The average squared differences between predictions and actual values. Squaring the errors removes negative errors. It also amplifies outliers (samples that have larger errors).

- For your regression models, you'll want to maximize R^2, whilst minimizing MAE and MSE.










