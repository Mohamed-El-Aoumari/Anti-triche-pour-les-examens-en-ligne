######################################################
##  Dans cette etape, on va creer une classe pour   ##
##  detecter la presence de telephone en utilisant  ## 
##                  YOLOv7 et OpenCV.               ##
######################################################


'''
Les caracteristiques du telephone extraites sont :
    ** phone_present : Integer (Binary), Indicator: 1 if a phone is detected, 0 otherwise.
    ** phone_loc_x, phone_loc_y : Float, X, Y coordinates of the detected phone. Zero if not detected.
    ** phone_conf : Float, Confidence score (0-1) of the phone detection.
'''

######################################################
## La sortie sera un dictionnaire de cette format : ##
## {                                                ##
##    'phone_present': 1,                           ##
##    'phone_loc_x': 150.0,                         ##
##    'phone_loc_y': 200.0,                         ##
##    'phone_conf': 0.85                            ##
## }                                                ##
######################################################


class PhoneDetectionExtractor:
    def __init__(self):
        pass

    def extract_phone_features(self, image):
        pass