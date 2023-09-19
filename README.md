# Face_mask_detection
* This project is for detecting a face mask if weared by a person or not
* It can be used for survelience in buildings or public areas during the time of pandemic like covid-19

### Note: This project is trained and built on google colab using google drive, therefore all the links, like path of directory, are written with respect to drive and you have to change these path according to your platform where you are implementing this project
  
##Techstack used:
* Yolov5
* Scikit-learn
* Pandas
* opencv-python
* PIL
* os
* matplotlib
### Installing requirements:
>!pip install -qr equirements.txt
### Dataset Link:
https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
### Importing Yolov5:
>!git clone https://github.com/ultralytics/yolov5

https://github.com/ultralytics/yolov5

### Overview of the dataset:
our dataset consist of images in .png format and their respective .xml file consists of the bounding boxes, names, labels, etc which will be required for training our model

## Preprocessing the data
For preprocessing the data, we need to convert the .xml files to new text files containing all the necessary informations like bounding boxes, labels, etc<br/>
while processing the images, we need to convert them into equal shapes to feed them into our model and their data has to be changes according to the change in the image

### For training the model, we use the following code as:
>cd yolov5
>python train.py --img 640 --batch 16 --epochs 100 --data ../mask_config.yaml --weights yolov5s.pt --workers 0

###  Testing
After the model is trained on the data, its time to test it over the test data.<br/>
This test data can be of any form like image, video, youtube link, etc








