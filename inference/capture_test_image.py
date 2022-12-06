import cv2
import os

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0
path="/Users/sharadc/Documents/uic/extra/repos/archive";
def hand_area(img):
    height, width,_ = img.shape;
    hand = img[4*height//10:8*height//10, 4*width//10:7*width//10]
    #hand = cv2.resize(hand, (378,378))
    return hand
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    #frame = cv2.flip(frame, 1)
    img =  hand_area(frame);
    cv2.imshow("test", img)

    k = cv2.waitKey(1)
    if k%256 == 27:
        break
    elif k%256 == 32:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(os.path.join(path,img_name), img)
        img_counter += 1
cam.release()

cv2.destroyAllWindows()