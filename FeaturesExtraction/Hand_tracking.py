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


class HandTrackingExtractor:
    def __init__(self):
        pass

    def extract_hand_features(self, image):
        pass