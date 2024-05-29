# Final Project: Detecting Good vs Bad Jumps

## Divakar Borra, Juan Sanchez Roa, Amira Garba

## Overview
 The objective of this project was to develop a mobile health sensing application that utilizes an accelerometer to record jump data and evaluate the quality of each jump. This application aims to analyze the user's jump technique during various fitness activities. We conducted jump quality recordings from Divakar, with approximately 5-minute sessions for each activity. To collect this data, we utilized the accelerometer on our iPhones and applied a combination of the Butterworth and Moving Average filters to eliminate excessive noise.  

  After careful analysis, we determined that the Butterworth filter was the most suitable option, as it effectively highlighted the prominent features in the graphs. We then identified that the peaks in the graph, specifically those associated with landing, played a crucial role in jump assessment. The difference in height between these peaks was used to classify jumps as either good or bad. Additionally, we derived the variance, mean, and entropy from the recorded sessions.  

  To train the data and create a decision tree, we employed these derived features. We considered the complexity of the data and the required accuracy when selecting the model. To evaluate the model's performance, we utilized appropriate measures such as accuracy, precision, and recall during the model validation process. Through an iterative refining procedure, we significantly improved the model's performance. We achieved this by adjusting the model's architecture, fine-tuning hyperparameters, and incorporating additional features or approaches.  

  Thanks to the iterative refinement, we were able to continuously enhance and optimize the model's performance. This iterative process led to improved accuracy and robustness in jump detection.  



Features used: 
- computing magnitude 
- computing mean
- computing variance 
- computing fft 
- computing entropy 
- counting peaks 


## Steps Required to Run
1. Run data-filtering.py to filter and import the raw data
2. Run data-labeling.py to combine the data
3. Run activity-classification-train.py
4. [optional] Run sensor-logger.py to classify data live (be connected with sensor logger app)
5. Jump 
### To visualize the Data
6. Run visual.ipynb to visualize data

## Known Issues
 ### `sensor-logger.py`:
  The accuracy of classifying live jumps are not as accurate as we wanted it to be. We believe that a few factors one reason would be there is a slight mistake in our code or whether we need more data to train and test. We also could not really figure out how to smooth out the live data for the classifier.
### *Confusion Matrix*:
  Although the confusion matrix works, it is not accurate as we want it to be. This is due to the fact that we might not have used the correct features or collect enough data. 