# skin-cancer-challenge-problem

# Kerollos Lowandy 

 **Repository: skin-cancer-challenge-problem**

# GitHub Link - https://github.com/Kerollosl/skin-cancer-challenge-problem


##CHOSEN MODEL:
VGG16

## NECESSARY PACKAGES:
- pandas: 2.1.3
- tensorflow: 2.15.0
- numpy: 1.26.2
- cv2: 4.9.0
- matplotlib: 3.8.2
- sklearn: 1.3.2


## DIRECTIONS:

1. Download directory from GitHub repository containing all necessary files.

2. Once in the directory, run "main.py" to run the CNN end to end process. This will train a model with the VGG16 architecture. If another architecture or method is desired, navigate to line 9 of "main.py" and change the hardcoded choice to the new desired choice.

3. Four images will show to clarify that each subset of the main dataset is functional.   
   They will need to be exited out of for the program to continue to run.

4. A confusion matrix will be generated based on the predictions and can either be saved or exited out of for the rest of the program to continue.

5. Once the main program has completed, the CSV needed to run the "threshold_shift.py" script will be ready. This script will run with the new binary prediction threshold of 0.45. If another threshold is desired, navigate to line 13 of "threshold_shift.py" and change the hardcoded choice to the new desired choice.


## CONTENTS:


"main.py" - Main program which preprocesses the raw data, trains the selected model architecture/training method, plots the predictions in a confusion matrix, and saves the model file.

"functions.py" - Used for organizational purposes to keep function definitions outside of main

"keras_tuner_script.py" - Used to define keras tuning algorithm

"threshold_shift.py" - Shifts the binary prediction threshold from 0.5 to a new desired threshold of choice and then replots a new confusion matrix based on that threshold. This requires main.py to run first in order for it to reference the generated CSV file

