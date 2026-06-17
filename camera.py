import cv2
import mediapipe as mp


class Camera:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def get_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            return None, None


        # flip camera
        frame = cv2.flip(frame, 1)


        # BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        return frame, mp_image


    def release(self):

        self.cap.release()