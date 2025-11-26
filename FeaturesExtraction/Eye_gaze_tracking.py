##################################################
##    Dans cette etape, on va creer une classe  ##
## pour extraire les caracteristiques du regard ##
##   des yeux a partir des images en utilisant  ##
##    MediaPipe Face Mesh, OpenCV et NumPy.     ##
##################################################

'''
Les caracteristiques du regard des yeux extraites sont :
    ** gaze_on_script : Integer (Binary), Indicator: 1 if gaze is directed towards the defined 'script' area, 0 otherwise.
    ** gaze_direction : Categorical, General direction of gaze. 'center', 'bottom_right', 'bottom_left', 'None'
    ** gazePoint_x, gazePoint_y : Float, X, Y coordinates of the estimated gaze point on the screen/frame.
    ** pupil_left_x, pupil_left_y : Float, X, Y coordinates of the left pupil.
    ** pupil_right_x, pupil_right_y : Float, X, Y coordinates of the right pupil.
'''

######################################################
## La sortie sera un dictionnaire de cette format : ##
## {                                                ##
##    'gaze_on_script': 1,                          ##
##    'gaze_direction': 'center',                   ##
##    'gazePoint_x': 400.0,                         ##
##    'gazePoint_y': 300.0,                         ##
##    'pupil_left_x': 180.0,                        ##
##    'pupil_left_y': 220.0,                        ##
##    'pupil_right_x': 220.0,                       ##
##    'pupil_right_y': 220.0                        ##
## }                                                ##
######################################################

class EyeGazeExtractor:
    def __init__(self):
        pass

    def extract_eye_gaze(self, image):
        pass

