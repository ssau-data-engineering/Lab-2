#Deriving the latest base image
FROM tensorflow/tensorflow:latest

RUN pip install scikit-learn numpy pandas fpdf

#Labels as key value pair
LABEL Maintainer="melik163.tensorflow_sklearn"


# Any working directory can be chosen as per choice like '/' or '/home' etc
# i have chosen /usr/app/src
WORKDIR /usr/app/src