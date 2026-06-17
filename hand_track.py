import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision



class HandTracker:


    def __init__(self):

        model_path = "hand_landmarker_model.task"


        base_options = python.BaseOptions(
            model_asset_path=model_path
        )


        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2
        )


        self.landmarker = (
            vision.HandLandmarker
            .create_from_options(options)
        )



    def detect(self, mp_image, frame):


        result = self.landmarker.detect(
            mp_image
        )


        if result.hand_landmarks:


            for hand in result.hand_landmarks:


                for landmark in hand:


                    h,w,_ = frame.shape


                    x = int(
                        landmark.x * w
                    )

                    y = int(
                        landmark.y * h
                    )


                    cv2.circle(
                        frame,
                        (x,y),
                        4,
                        (255,0,0),
                        -1
                    )


                    print(
                        "x:",
                        landmark.x,
                        "y:",
                        landmark.y,
                        "z:",
                        landmark.z
                    )


        return frame
    
    