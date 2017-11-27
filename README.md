# ECE6258_CNNevaluation

This package is designed to achieve two different objectives :
1. Provide real time object recognition and location for environmental awareness
2. Provide an analysis of different Convolutional Neural Netwoorks (CNN) performances with respect to distortions

For your confort, this file is available in .md or .html.

## Introduction
Developed by Sylvain Chatel for ECE 6258 project, this code is divided into four directories. Each of them has a particular purpose. ```Database/``` aims at retrieving and distorting a database of images. ```mobileNet/```is the directory that holds the real-time image recognition code along with the code for batch recognition. This code was inspired and modified from several online resources such as OpenCV, PYImageSearch and tensorflow tutorial. The ```Darknet/```directory is a clone of Joseph Redmon git : ```https://github.com/pjreddie/darknet.git```. This package provides a code to launch his CNNs called darknet and YOLO. Finally ```Analysis/```provide a analysis of the different CNN performances.

Note : The most interesting part of this work lies in the Analysis and not in the CNNs implementation.

## Note 
Every commands were launched from the terminal. Please make sure you are in the rigth directory before launching.

## Dependencies and requirements
This code was tested on OSX 10.11.6 using python 2.7. The machine had 8GB 1867 MHz memory and a 2.7 GHz Intel Core i5 processor. We decided not to run it on the GPU in order to keep in mind that in our project mindset, this software should be run on edge devices with no GPU resources. 

The following packages need to be installed on the machine : 
- OpenCV 
- python 2.7
- matplotlib
- tabulate
- numpy
- pickle
- os
- argparse
- Matlab 2017a

The remaining packages where installed and put inside directories directly in the workspace so there should be no need to download it again.
Please note also that real-time detection requires a webcam to access a live feed. 

## Plan
1. Workspace overview
2. Real-time recognition
3. Database creation
4. CNN evaluation
5. Analysis
6. Conclusion
7. Scripts overview

## 1. Workspace overview
Analysis/  
- analyse_all.py : python script comparing all CNNs in pickles/ with respect to accuracy and plot results  
- analyse.py : python script determinining a CNN performance with respect to time, accuracy and distortions  
- csv2pickle.py : python script creating a python pickle variable from a csv for YOLO and mobileNet CNN  
- csv2pickle_2.py :  python script creating a python pickle variable from a csv for all other CNNs 
- confusion.py : python script to compute confusion matrices

Analysis/CNN_csv/ :  directory holding the csv  

Analysis/pickles/ :  directory holding the pickle variables  


darknet/ : Darknet package from ```https://github.com/pjreddie/darknet.git```  


Database/ :  Directory holding the image database  


Matlab/ :  Directory holding ```main.m```to generate distortions  

mobileNet/ 
mobileNet/object_detect/  
- main_object_det.py :  python script to detect object from batch of images using MobileNet SSD CNN  
- mobile_dist.txt :  Example of output of ```main_obj_det.py``` 
- object_detection2.py : python script to detect from single image using MobileNet SSD CNN  
- MobileNetSSD_deploy.caffemodel : Config file  
- MobileNetSSD_deploy.prototxt.txt : Config file 

mobileNet/object_detect/image : Directory for test images  
mobileNet/RT_object_detect/    
- MobileNetSSD_deploy.caffemodel : Config file  
- MobileNetSSD_deploy.prototxt.txt : Config file  
- real_time_object_detection2.py : python script for real-time detection


OpenCV/ : OpenCV package from github   
README.md

## 2. Real-time recognition
In order to run a real-time detection using a webcam and MobileNet SSD CNN, the user needs to go to ```mobileNet/RT_object_detect/``` and run ```real_time_object_detection2.py```.

```
python real_time_object_detection2.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
```

To run the network over a single image, go to ```mobileNet/object_detect/``` and run ```object_detection2.py```.

```
python object_detection2.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --image images/example_1.jpg
```
Finally, to run the CNN over a batch of images, specify the directory with ```--directory``` and run ```main_obj_det.py```.

```
python main_obj_det.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --directory images
```
This will generate a .txt file similar to mobile_dist.txt.


### Note : 
When running the real-time script, the webcam will be used as the input stream. Objects such as chairs, table, sofas and more will be detected. We have modified the code so that if a chair/table/sofa is on the users path (2 seconds delay) a warning message will be printed in the console. In the end this was design to trigger an alarm for the user such as "warning : chair in front of you".

## 3. Database creation
In order to generate the image database, we used Imagenet database. We bounded ourselves to three classes : {chair}, {table} and {sofa}. Namely those were respectively databases n03001627, n03201208 and n04256520 from imagenet. After creating an Imagenet account, just find the database fitting you rneeds and download the database tar. 

Then untar it using ```tar xvzf data.tar.gz```.

Now that a directory of the database is created, we can generate distortions. Using ```main.m``` in the Matlab directory, a user can generate equalized images, gaussian blurred images (with coefficient 2, 5 and 7), two darkened and one lightned image. To do so, just move ```main.m``` in the same directory as the database and run it. *Note: user might want to keep only .JPEG files in the database directory*. This will generae the altered database in the directory. 

This was done using Matlab 2017a.

Now you can generate a .txt file with the name of all images in this database. This will be usefull later on to run on several CNNs. To do so, we will use the strea editor ```sed```.

```
dir data_distorted | tr ' ' '\n' | sed '/^$/d' | tr -d ' ' | sed 's/^/data_distorted\//;' > dist_img.txt
```
Note that white space needs to be fixed. Using the vim tools I had I just ran ```:FixWhitespace```.

## 4. CNN evaluation
After generating the altered database for all three classes, you can marged them into a new directory : ```data_distorted/```. 

Now we can run the different CNNs in order to generate csv.txt files. 

The work developed by Joseph Redmon ```https://pjreddie.com/```and ```https://github.com/pjreddie/darknet.git``` makes possible to run CNNs he designed and some of the state of the art CNNs. *Note: there is no information however on the trainning used for the learning*.

This phase can take more or less time (between seconds to more than 26 hours depending on the CNN and the dataset). Hence we propose these command which run on a subset of the distorted database located in Database/data_test or in darknet/test2.txt.

### MobileNet SSD
Go to mobileNet/object_detect/ directory.

```
python main_obj_det.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --directory ~/python6258/Database/data_test  -c 0.2 >mobile_test.txt
```

Then, to export this csv.txt to a pickle variable just copy the .txt to Analysis/ and run :
```
python csv2pickle.py -f mobile_test.txt - mobile_test
```

### tiny YOLO using ./darknet
Go to darknet/ directory.

```
./darknet detector test cfg/voc.data cfg/tiny-yolo-voc.cfg weights/tiny-yolo-voc.weights < test.txt | sed 's/data_distorted\///;  s/: Predicted in /\;/;s/ seconds./\;/; s/\n/\;/; s/: /\;/; s/%/\;/ ' | tr '\n' ' ' | tr -d ' ' | sed 's/EnterImagePath;//;' | sed 's/EnterImagePath;/\n/g ' >tyolo_dist.txt
```

Then, to export this csv.txt to a pickle variable just copy the .txt to Analysis/ and run :
```
python csv2pickle.py -f tyolo_dist.txt - tyolo_test
```

### densenet201 
Go to darknet/ directory.

```
./darknet classifier predict cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights < test_dist.txt |& tee densenet_test1.txt
```
Make the .txt look better in stream editor

```
cat densenet_test1.txt| sed 's/^/\;/' |  sed 's/data_distorted\///;  s/: Predicted in /\;/;s/ seconds./\;/; s/\n/\;/; s/: /\;/; s/%// ' | tr -d '\n'  | sed 's/;Enter Image Path\;//;' | sed 's/Enter Image Path\;/\n/g' | sed 's/\;\;/\;/;'  >densenet_test.txt
```

Then, to export this csv.txt to a pickle variable just copy the .txt to Analysis/ and run :
```
python csv2pickle_2.py -f densenet_test.txt - densenet_test
```

### darknet 
Go to darknet/ directory.

```
./darknet classifier predict cfg/imagenet1k.data cfg/darknet.cfg weights/darknet.weights < test_dist.txt |& tee darknet_test1.txt
```
Make the .txt look better in stream editor

```
cat darknet_test1.txt| sed 's/^/\;/' |  sed 's/data_distorted\///;  s/: Predicted in /\;/;s/ seconds./\;/; s/\n/\;/; s/: /\;/; s/%// ' | tr -d '\n'  | sed 's/;Enter Image Path\;//;' | sed 's/Enter Image Path\;/\n/g' | sed 's/\;\;/\;/;'  >darknet_test.txt
```

Then, to export this csv.txt to a pickle variable just copy the .txt to Analysis/ and run :
```
python csv2pickle_2.py -f darknet_test.txt - darknet_test
```

### darknet19
Go to darknet/ directory.

```
./darknet classifier predict cfg/imagenet1k.data cfg/darknet19.cfg weights/darknet19.weights < test_dist.txt |& tee darknet19_test1.txt
```
Make the .txt look better in stream editor

```
cat darknet19_test1.txt| sed 's/^/\;/' |  sed 's/data_distorted\///;  s/: Predicted in /\;/;s/ seconds./\;/; s/\n/\;/; s/: /\;/; s/%// ' | tr -d '\n'  | sed 's/;Enter Image Path\;//;' | sed 's/Enter Image Path\;/\n/g' | sed 's/\;\;/\;/;'  >darknet19_test.txt
```

Then, to export this csv.txt to a pickle variable just copy the .txt to Analysis/ and run :
```
python csv2pickle_2.py -f darknet19_test.txt - darknet19_test
```


## 5. Analysis

## 6. Conclusion

## 7. Scripts overview

### real_time_object_detection2.py - adapted
### object_detection2.py -adapted
### main_obj_det.py
### csv2pickle.py
### csv2pickle_2.py
### ./darknet from J.Redmon
### analyse.py
### analyse_all.py
### confusion.py
