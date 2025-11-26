########################################################
##   Dans cette etape, on va creer une classe pour    ##
## extraire les caracteristiques de la main a partir  ##
## des images en utilisant MediaPipe Hands et OpenCV. ##
########################################################

'''
Les caracteristiques de la main extraites sont :
    ** hand_count : Integer, Number of hands detected in the frame.
    ** left_hand_x, left_hand_y : Float, X, Y coordinates of the left hand. Zero if not detected.
    ** right_hand_x, right_hand_y : Float, X, Y coordinates of the right hand. Zero if not detected.
    ** hand_obj_interaction : Integer (Binary), Indicator: 1 if hand is interacting with an object, 0 otherwise.
'''

#######################################################
## La sortie sera un dictionnaire de cette format :  ##
## {                                                 ##
##    'hand_count': 2,                               ##
##    'left_hand_x': 120.0,                          ##
##    'left_hand_y': 250.0,                          ##
##    'right_hand_x': 300.0,                         ##
##    'right_hand_y': 260.0,                         ##
##    'hand_obj_interaction': 1                      ##
## }                                                 ##
#######################################################

class HandTrackingExtractor:
    def __init__(self):
        pass

    def extract_hand_features(self, image):
        pass