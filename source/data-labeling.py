
"""
This is the script used to combine all collected csv data files into
a single csv file.
"""

import numpy as np
import csv
import time

import labels


# print the available class label (see labels.py)
act_labels = labels.activity_labels
print(act_labels)

# specify the data files and corresponding activity label
# csv_files = ["data/Accelerometer_GoodJump1.csv",
#              "data/Accelerometer_GoodJump2.csv",
#              "data/Accelerometer_GoodJump3.csv",
#              "data/Accelerometer_BadJump1.csv",
#              "data/Accelerometer_BadJump2.csv",
#              "data/Accelerometer_Standing1.csv",
#              "data/Accelerometer_Standing2.csv"
#              ]

csv_files = ["filtered-data/filtered-GoodJump1.csv",
             "filtered-data/filtered-GoodJump2.csv",
             "filtered-data/filtered-GoodJump3.csv",
             "filtered-data/filtered-BadJump1.csv",
             "filtered-data/filtered-BadJump2.csv",
             "filtered-data/filtered-Standing1.csv",
             "filtered-data/filtered-Standing2.csv"
             ]

activity_list = ["Good Jump",
                 "Good Jump",
                 "Good Jump",
                 "Bad Jump",
                 "Bad Jump",
                 "Standing",
                 "Standing"]

# Specify final output file name. 
output_filename = "data/all_labeled_data.csv"


all_data = []

zip_list = zip(csv_files, activity_list)

for f_name, act in zip_list:

    if act in act_labels:
        label_id = act_labels.index(act)
    else:
        print("Label: " + act + " NOT in the activity label list! Check label.py")
        exit()
    print("Process file: " + f_name + " and assign label: " + act + " with label id: " + str(label_id))

    with open(f_name, "r") as f:
        reader = csv.reader(f, delimiter = ",")
        #headings = next(reader)
        for row in reader:
            row.append(str(label_id))
            all_data.append(row)


with open(output_filename, 'w',  newline='') as f:
    writer = csv.writer(f)
    writer.writerows(all_data)
    print("Data saved to: " + output_filename)