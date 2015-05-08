# face-detection-gabor
Face Detection using Neural Networks and Gabor Features

# To run:

    1- copy all files and directories to the MATLAB's work folder
     * In order to run the program you must have Image Processing Toolbox

	2- (Important) Navigate to the root folder which contains "main.m".

    3- Type "main" or "run main" in the command window

	4. Only fort the first time, the program creates Gabor filters and stores them in ./data/gabor.mat Training set dataset and stores it in ./data/imgdb.mat Neural Network and stores it in ./data/net.mat
		
	5- imgdb is short for "image data base".
	
    6- When the program menu appears click on "Train Network" and wait until the program is done with the training

    7- click on "Test on Photos". A file dialog appeares. Select a .jpg photo. The selected photo may contain several face images but the program can only detect faces that can be fit inside a 27x18 window.
    
    8- Wait until the program detects some faces.

Note*:
1) This program only detects faces that can fit inside a 27x18 window.
2) This program requires Image Processing Toolbox
3) This program is tested on MATLAB 2013a and probably will not work with
   older versions of MATLAB.   
	
For more information please visit the website. Thank you.

http://www.facedetectioncode.com