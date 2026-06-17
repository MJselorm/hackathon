import cv2

from camera import Camera
from hand_track import HandTracker



def main():


    camera = Camera()

    tracker = HandTracker()



    while True:


        frame, mp_image = camera.get_frame()


        if frame is None:
            break



        frame = tracker.detect(
            mp_image,
            frame
        )



        cv2.imshow(
            "Hand Detection",
            frame
        )


        if cv2.waitKey(1) & 0xFF == 27:
            break



    camera.release()

    cv2.destroyAllWindows()



if __name__ == "__main__":

    main()