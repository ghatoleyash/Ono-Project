## Video to Normal-Anomaly Frames

#### Functions:
- Data generation takes video as an input and gives individual frames as an output
- Functions to segregate the normal as well as anomalous frames which are mentioned in the [AnomalyLabel.xlsx](https://github.com/ghatoleyash/Ono-Project/blob/main/Codes/Data%20Generation/AnomalyLabel.xlsx) in the anomaly sheet
- Moreover, it can skip the frames to be saved or in other words play the video in 2x, 3x,... and save the frames accordingly

Click [here](https://github.com/ghatoleyash/Ono-Project/blob/main/Codes/Data%20Generation/videoToAnomaly.py) to check in-detail implementation and follow this file to run the code


#### Libraries to Install:
```
pip install opencv-python
pip install opencv-contrib-python
pip install numpy
pip install pandas
```

#### How to Run:
- create an anomaly xlsx file with sheet name as "Anomaly"
- In this sheet, add three columns video name, start_time, end_time (these are start and end time of anomaly occurrence)
- Replace the "anomalycsv" variable in [videoToAnomaly.py](https://github.com/ghatoleyash/Ono-Project/blob/main/Codes/Data%20Generation/videoToAnomaly.py) with the path of above xlsx file
- Change the variable name "path1" in [videoToAnomaly.py](https://github.com/ghatoleyash/Ono-Project/blob/main/Codes/Data%20Generation/videoToAnomaly.py) to where the video resides
- "path2" and "AnomalyPath" can be specified by the user in order to save the frame-data at those locations
- "speed" this variable determines the playbackspeed of the video if set as 2 it means video will run in 2x, or in other words it will save every alternate frame, so on and so forth
- Lastly, run the below command
```
python videoToAnomaly.py
```


#### Sample Frame-by-Frame Images 
<img src="https://github.com/ghatoleyash/Ono-Project/blob/main/Images/00.jpg" width=80% height=50%>

#### Data Segregation
<img src="https://github.com/ghatoleyash/Ono-Project/blob/main/Images/Data_Segregation.png" width=80% height=50%>


