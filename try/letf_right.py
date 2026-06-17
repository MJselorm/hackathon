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


        self.left_hand = None
        self.right_hand = None



    # Detect hands
    def detect(self, frame):


        # Mirror camera
        frame = cv2.flip(frame, 1)



        # BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )



        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )



        result = self.landmarker.detect(
            mp_image
        )



        self.left_hand = None
        self.right_hand = None



        if result.hand_landmarks:


            for i, hand in enumerate(
                result.hand_landmarks
            ):


                hand_type = (
                    result.handedness[i][0].category_name
                )



                if hand_type == "Left":

                    self.left_hand = hand


                elif hand_type == "Right":

                    self.right_hand = hand




                # Draw points
                for landmark in hand:


                    h, w, _ = frame.shape


                    x = int(
                        landmark.x * w
                    )


                    y = int(
                        landmark.y * h
                    )


                    cv2.circle(
                        frame,
                        (x,y),
                        5,
                        (255,0,0),
                        -1
                    )



        return frame





    # Get left hand
    def get_left_hand(self):

        return self.left_hand





    # Get right hand
    def get_right_hand(self):

        return self.right_hand






    # Count fingers
    def count_fingers(self, hand):


        if hand is None:

            return 0



        count = 0



        # determine left/right

        if hand == self.left_hand:

            is_left = True

        else:

            is_left = False




        # Thumb

        if is_left:

            if hand[4].x > hand[3].x:
                count += 1

        else:

            if hand[4].x < hand[3].x:
                count += 1




        # Index

        if hand[8].y < hand[6].y:

            count += 1



        # Middle

        if hand[12].y < hand[10].y:

            count += 1



        # Ring

        if hand[16].y < hand[14].y:

            count += 1




        # Pinky

        if hand[20].y < hand[18].y:

            count += 1



        return count








# =========================
# TEST RUN
# =========================

if __name__ == "__main__":


    tracker = HandTracker()


    cam = cv2.VideoCapture(0)



    while cam.isOpened():


        ret, frame = cam.read()


        if not ret:

            break




        frame = tracker.detect(
            frame
        )



        left = tracker.get_left_hand()

        right = tracker.get_right_hand()



        print(
            "Left:",
            tracker.count_fingers(left),
            "| Right:",
            tracker.count_fingers(right)
        )




        cv2.imshow(
            "Hand Tracking",
            frame
        )




        if cv2.waitKey(1) & 0xFF == 27:

            break




    cam.release()

    cv2.destroyAllWindows()