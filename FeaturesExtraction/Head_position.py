########################################################
##  Dans cette etape, on va creer une classe pour     ##
##    extraire les caracteristiques de la tete a      ##
## partir des images en utilisant MediaPipe Face Mesh,## 
##               OpenCV et NumPy.                     ##
########################################################

'''
Les caracteristiques de la tete extraites sont :
    ** head_pose : Categorical: 'forward', 'down', 'left', 'right', 'None'
    ** head_pitch : Float, Head rotation around the X-axis in degrees.
    ** head_roll : Float, Head rotation around the Z-axis (tilt left/right).
    ** head_yaw : Float, Head rotation around the Y-axis (left/right turn).
'''

########################################################
## La sortie sera un dictionnaire de cette format :   ##
## {                                                  ##
##    'head_pose': 'forward',                         ##
##    'head_pitch': 10.5,                             ##
##    'head_roll': -5.2,                              ##
##    'head_yaw': 15.0                                ##
## }                                                  ##
########################################################

class HeadPositionExtractor:
    def __init__(self):
        pass

    def extract_head_position(self, image):
        pass