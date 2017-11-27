# ECE6258_CNNevaluation

This package is designed to achieve two different objectives : 
1. Provide real time object recognition and location for environmental awareness 
2. Provide an analysis of different Convolutional Neural Netwoorks (CNN) performances with respect to distortions 

For your confort, this file is available in .md or .html.

## Introduction
Developed by Sylvain Chatel for ECE 6258 project, this code is divided into four directories. Each of them has a particular purpose. ```Database/``` aims at retrieving and distorting a database of images. ```mobileNet/``` is the directory that holds the real-time image recognition code along with the code for batch recognition. This code was inspired and modified from several online resources such as OpenCV, PYImageSearch and tensorflow tutorial. The ```Darknet/``` directory is a clone of Joseph Redmon git : ```https://github.com/pjreddie/darknet.git```. This package provides a code to launch his CNNs called darknet and YOLO. Finally ```Analysis/``` provides a analysis of the different CNN performances.

Note : The most interesting part of this work lies in the Analysis and not in the CNNs implementation.

## Note 
Every commands were launched from the terminal. Please make sure you are in the right directory before launching.

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
python object_detection2.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --image images/example_1.JPEG
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
python csv2pickle.py -f mobile_test.txt -o mobile_test
```

### tiny YOLO using ./darknet
Go to darknet/ directory. Please note that aside from a few text files and database directories this darknet/directory was retrieve from J.Redmon Github and was used as is.

```
./darknet detector test cfg/voc.data cfg/tiny-yolo-voc.cfg weights/tiny-yolo-voc.weights < test2.txt | sed 's/data\///;  s/: Predicted in /\;/;s/ seconds./\;/; s/\n/\;/; s/: /\;/; s/%/\;/ ' | tr '\n' ' ' | tr -d ' ' | sed 's/EnterImagePath;//;' | sed 's/EnterImagePath;/\n/g ' >tyolo_test.txt
```

Then, to export this csv.txt to a pickle variable just copy the .txt to Analysis/ and run from there:
```
python csv2pickle.py -f tyolo_test.txt -o tyolo_test
```

### densenet201 
Go to darknet/ directory.

```
./darknet classifier predict cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights < test2.txt |& tee densenet_test.txt
```
Make the .txt look better in stream editor

```
cat densenet_test1.txt| sed 's/^/\;/' |  sed 's/data\///;  s/: Predicted in /\;/;s/ seconds./\;/; s/\n/\;/; s/: /\;/; s/%// ' | tr -d '\n'  | sed 's/;Enter Image Path\;//;' | sed 's/Enter Image Path\;/\n/g' | sed 's/\;\;/\;/;'  >densenet_test.txt
```

Then, to export this csv.txt to a pickle variable just copy the .txt to Analysis/ and run from there:
```
python csv2pickle_2.py -f densenet_test.txt -o densenet_test
```

### darknet 
Go to darknet/ directory.

```
./darknet classifier predict cfg/imagenet1k.data cfg/darknet.cfg weights/darknet.weights < test2.txt |& tee darknet_test.txt
```
Make the .txt look better in stream editor after removing manually the network model (until the first 'Enter'): 

```
cat darknet_test1.txt| sed 's/^/\;/' |  sed 's/data\///;  s/: Predicted in /\;/;s/ seconds./\;/; s/\n/\;/; s/: /\;/; s/%// ' | tr -d '\n'  | sed 's/;Enter Image Path\;//;' | sed 's/Enter Image Path\;/\n/g' | sed 's/\;\;/\;/;'  >darknet_test.txt
```

Then, to export this csv.txt to a pickle variable just copy the .txt to Analysis/ and run from there:
```
python csv2pickle_2.py -f darknet_test.txt -o darknet_test
```

### darknet19
Go to darknet/ directory.

```
./darknet classifier predict cfg/imagenet1k.data cfg/darknet19.cfg weights/darknet19.weights < test2.txt |& tee darknet19_test.txt
```
Make the .txt look better in stream editor

```
cat darknet19_test1.txt| sed 's/^/\;/' |  sed 's/data\///;  s/: Predicted in /\;/;s/ seconds./\;/; s/\n/\;/; s/: /\;/; s/%// ' | tr -d '\n'  | sed 's/;Enter Image Path\;//;' | sed 's/Enter Image Path\;/\n/g' | sed 's/\;\;/\;/;'  >darknet19_test.txt
```

Then, to export this csv.txt to a pickle variable just copy the .txt to Analysis/ and run :
```
python csv2pickle_2.py -f darknet19_test.txt -o darknet19_test
```


## 5. Analysis
Now that we have generated a database of image recognition for different CNNs when different distortions are applied, we can proceed to the analysis. Please change the current folder to the Analysis/ directory. 

In this directory, the reader can find five scripts and two directories. The first two scripts ```csv2pickle.py``` and ```csv2pickle_2.py``` are designed to export the csv generated in the previous section to a python pickle variable. Depending on the CNN used, one of the two scripts is to be used. This is due to a difference in the output provided by ./Darknet. 

If the CNN is MobileNet or Tiny Yolo, please use ```csv2pickle.py```. Else, use the other one. 

The generated pickles can be moved to the ```pickles/``` directory for future use.  

The original CSV files are stored in the ```CNN_csv/```directory.  


At this point, the analysis can start. To do so, we created different scripts to help go through the database.  

### analyse.py
This versatile script is the beginning of everything. It can display scatter plots of the elapsed time or the accuracy for different filters (object class speciifed or distortion specified). Let us expand a little more on how it works. 

```
python analyse.py -f pickles/arg0 -p arg1 -feat arg2 -obj arg3 -dist arg4 -rel arg5 -l arg6 -r arg7
```

arg0 is the name of the pickle file with the pickle extension -- e.g. 'tinyyolo_dist.pickle'.  
arg1 is a string to specified if user wants to display plots or not -- 'yes' /'no'.  
arg2 is the feature user wants to analyse : choose between 'time', 'accuracy' or 'comparison'.  
arg3 is the object class filter : '0' for chairs, '1' for table and '2' for sofas.  
arg4 is the distortion filter : '0' for the original, '1' for the equalized, '2' for the blurred 2, '3' for the blurred 5, '4' for the blurred 7, '5' for the ligthened, '6' for the very lightened and '7' for the darkened.  
arg5 is the knob to proceed to the 'comparison' feature for accuracy relative to original images.  
arg6 is the knob to limit the number of samples considered -- e.g. -l 10 takes the first ten images.   
arg7 is to set the previous selection to a rndom pooling instead of taking the first images.  

Let us present some examples 
```
python analyse.py -f pickles/mobile_dist.pickle -p yes -feat time
```
Displays a scatter plot of elapsed time for all distortions for all images for the MobileNet recognition. 

```
python analyse.py -f pickles/mobile_dist.pickle -p yes -feat time -obj 0 -dist 1
```
Displays a scatter plot of elapsed time for equalized chair images for the MobileNet recognition. 

```
python analyse.py -f pickles/mobile_dist.pickle -p yes -feat accuracy -obj 2 -dist 5
```
Displays a scatter plot of the accuracy along with statistics in the console for ligthened sofa images recognised with MobileNet. 

```
python analyse.py -f pickles/mobile_dist.pickle -p yes -feat comparison
```
Returns in the console the comparative table for MobileNet recognition with respect to distortions for all images from all three classes. If ```-p yes``` is set, then it will also plot in Figure 1 the candlestick analysis and in Figures 2 to 9 the histograms of the accuracy. 

We can observe that those histograms seem to be exponential. Also, the candlestick graph represent for each distortion the statistical results : the black line link the maximum accuracy and the minimum accuracy. The cross marks the mean accuracy and the grey box the accuracy plus and minu the standard deviation.  

```
python analyse.py -f pickles/mobile_dist.pickle -p yes -feat comparison -obj 0
```
This proceed to the same analysis expect the object are only taken from the 'chair' database. The dropped percentage represent the percentage of 'dropped' images : images that could no longer be recognised by the network as chairs. 

```
python analyse.py -f pickles/mobile_dist.pickle -p no -feat comparison -obj 0 -rel true
```
By setting the argument ```-rel true``` the statistical analysis is no longer made on the accuracy but the difference of accuracy with respect to the accuracy in the original image.  

### analyse_all.py
This script allows user to do an analysis over all CNNs at the same time. By putting as an argument the directory storing the pickle files, a statistical analysis can be conducted. It generates an analysis for all images in the database regardless of their class. Accuracy refers to the predominant object in the image -which is supposedly the object of from the class the image is from (chair, table or sofa). 

```
python analyse_all.py -d arg0 -p arg1 -rel arg2
```
arg0 is the directory storing the pickles. 
arg1 is to plot or not the graphs -- 'yes' /'no'. 
arg2 is to set the relative coparison. 

```
python analyse_all.py -d pickles -p yes
```
This creates Table 1 presented in the report (need to add manually the ellapsed time). It also plots a candlestick graph for all CNN for all distortions. 

```
python analyse_all.py -d pickles -p yes -rel true
```
Proceeds to the same analysis with a relative approach. 

### confusion.py
This script creates confusion matrices for a particular distortion for a particular CNN.
```
python confusion.py -f arg0 -dist arg1
```
arg0 is the path to the pickle file.  
arg1 is the distortion to use : '0' for the original, '1' for the equalized, '2' for the blurred 2, '3' for the blurred 5, '4' for the blurred 7, '5' for the ligthened, '6' for the very lightened and '7' for the darkened.  
 
This script creates confusion matrices for a particular distortion for a particular CNN.    

We can explicit a bit more on this matrix. Let say M=m(i,j), then m(i,j) is the probability of having an image from database i classified with a j predominant object.  

As an example, we can run this scrip for MobileNet with no distortion:
```
python confusion.py -f pickles/mobile_dist.pickle -dist 0
```
This script is quite useful to assess the effect of distortions on classification capabilities of CNNs.  

## 6. Conclusion
In this semester long project, we were able to use state of the art machine recognition based on CNN to produce a real-time environmental awareness software. Through an analysis of different CNNs, we were able to assess the resiliency and performances with respect to distortions that could be encountered in real life : blur, exposition or change in colors.

## 7. Scripts overview
Overview of how the funcions are working. 

### real_time_object_detection2.py - adapted
```
python real_time_object_detection2.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
```

### object_detection2.py -adapted
```
python object_detection2.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --image arg0
```
arg0 is the path to the image -e.g. ```images/example_1.JPEG```.

### main_obj_det.py
```
python main_obj_det.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --directory arg0
```
arg0 is the name of the directory storing image to classify.

### csv2pickle.py
```
python csv2pickle.py -f arg0 -o arg1
```
arg0 is the input csv file --e.g. CNN_csv/tyolo_dist.txt  
arg1 is the output pickle file name --e.g. tinyyolo_dist

### csv2pickle_2.py
```
python csv2pickle_2.py -f arg0 -o arg1
```
arg0 is the input csv file --e.g. CNN_csv/tyolo_dist.txt  
arg1 is the output pickle file name --e.g. tinyyolo_dist

### ./darknet from J.Redmon
Developed by J.Redmon, please see above for how to run it. We also added some ```sed```modifications in order to make the output more readable.

### analyse.py
```
python analyse.py -f pickles/arg0 -p arg1 -feat arg2 -obj arg3 -dist arg4 -rel arg5 -l arg6 -r arg7
```

arg0 is the name of the pickle file with the pickle extension -- e.g. 'tinyyolo_dist.pickle'.  
arg1 is a string to specified if user wants to display plots or not -- 'yes' /'no'.  
arg2 is the feature user wants to analyse : choose between 'time', 'accuracy' or 'comparison'.  
arg3 is the object class filter : '0' for chairs, '1' for table and '2' for sofas.  
arg4 is the distortion filter : '0' for the original, '1' for the equalized, '2' for the blurred 2, '3' for the blurred 5, '4' for the blurred 7, '5' for the ligthened, '6' for the very lightened and '7' for the darkened.  
arg5 is the knob to proceed to the 'comparison' feature for accuracy relative to original images.  
arg6 is the knob to limit the number of samples considered -- e.g. -l 10 takes the first ten images.   
arg7 is to set the previous selection to a rndom pooling instead of taking the first images.  


### analyse_all.py
```
python analyse_all.py -d arg0 -p arg1 -rel arg2
```
arg0 is the directory storing the pickles.  
arg1 is to plot or not the graphs -- 'yes' /'no'.  
arg2 is to set the relative coparison.  

### confusion.py
```
python confusion.py -f arg0 -dist arg1
```
arg0 is the path to the pickle file.  
arg1 is the distortion to use : '0' for the original, '1' for the equalized, '2' for the blurred 2, '3' for the blurred 5, '4' for the blurred 7, '5' for the ligthened, '6' for the very lightened and '7' for the darkened.  

