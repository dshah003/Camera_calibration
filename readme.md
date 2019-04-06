# CMSC 733 Home work 1  
## Auto-Camera Calibration  

### Requirements:  
1. OpenCV
2. numpy
3. scipy  

### To Calibrate camera:  
**Step 1:** Capture about 13 images of the checkerboard each in a different view/orientation
**Step 2:** Add these images to data folder in the working directory
**Step 3:** In the file wrapper.py set the SquareSize and PatternSize variables according to your checkerboard pattern.  
**Step 4:** Run the python code by running the following code in the terminal  
```console  
$ python wrapper.py  
```  

The program must output the intrinsic parameter in the terminal window.