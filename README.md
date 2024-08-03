# Medizintechnische_Systeme
# Dart Score Prediction Project

This repository contains two parts of a project focused on predicting dart scores using computer vision and machine learning techniques.
The first step is to pull this directory into your local PC.
## Installation

To install the necessary dependencies, you can use the `Requirements.txt` file provided. First create a new environment using the command:
"conda create --name name python=3.8"
and then activate it using the command:
"conda activate name "
Use the following command to install the dependencies:
"pip install -r Requirements.txt"

Once all requirements are installed you can test both parts of the project by:
1. open the GUI to predict scores from the dartboard (based on the deepdarts-d2 model) by running the following command:
"python bbox_GUI_modified.py"
The GUI gives you the possibility to load videos and predict their scores for videos containing a maximum of 3 Darts. It is also possible to open a live camera and make sure that your dartboard is visible and then predict the scores of the darts.

2. To run the GUI for Score prediction from the Player's arm motion you can run this command:
"python GUI_Predictor.py" 
and then you have the possibility to load a video of the player during the throwing motion. You can cut the video to only include the 2-3s of the exact throwing motion by clicking of 'yes' button when it shows after clicking on the "predict score" button, if your video is already cut you can choose 'No' .
You can also display the angles of the player's arm  during the throwing motion by clicking on the "visualize angles"
You also have the possibility to predict the score and do all the previous steps using the live camera .


