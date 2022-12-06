# import cv2
# import mediapipe as mp
# import time
#
# cap = cv2.VideoCapture(0)
#
# mpHands = mp.solutions.hands
# hands = mpHands.Hands()
# mpDraw = mp.solutions.drawing_utils
#
# pTime = 0
# cTime = 0
#
# while True:
#     success, img = cap.read()
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(imgRGB)
#     print(results.multi_hand_landmarks)
#
#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             for id, lm in enumerate(handLms.landmark):
#                 #print(id,lm)
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x *w), int(lm.y*h)
#                 #if id ==0:
#                 cv2.circle(img, (cx,cy), 7, (255,0,255), cv2.FILLED)
#
#             mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
#
#
#     cTime = time.time()
#     fps = 1/(cTime-pTime)
#     pTime = cTime
#
#     cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
#
#     cv2.imshow("Image", img)
#     cv2.waitKey(1)
import cv2
import mediapipe as mp
import time
import numpy as np;


def create_rectangle_around_hand(hands2DList, img, XS, YS):
    hands2DListnp = np.array(hands2DList);
    centroids_for_each_hand = [];
    opposite_points_for_each_hand = [];
    XS = 200;
    YS = 150;
    for i in range(0, hands2DListnp.shape[0]):
        handLandmarks = hands2DListnp[i];
        sum_row = np.sum(handLandmarks[:, 0]);
        sum_column = np.sum(handLandmarks[:, 1]);
        centroid = [sum_row / handLandmarks.shape[0], sum_column / handLandmarks.shape[0]];
        opposite_points = [[centroid[0] - XS, centroid[1] - YS], [centroid[0] + XS, centroid[1] + YS]];
        opposite_points_np = np.array(opposite_points);
        opposite_points_np_int = opposite_points_np.astype("int");
        print((opposite_points_np_int[0, 0], opposite_points_np_int[0, 1]));
        print((opposite_points_np_int[1, 0], opposite_points_np_int[1, 1]));
        cv2.rectangle(img, (opposite_points_np_int[0, 0], opposite_points_np_int[0, 1]),
                      (opposite_points_np_int[1, 0], opposite_points_np_int[1, 1]), (0, 255, 0), 3);
        centroids_for_each_hand.append(centroid);
        opposite_points_for_each_hand.append(opposite_points);


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

#while True:
#success, img = cap.read()
img = cv2.imread("../opencv_frame_1.png");
#img = cv2.imread("/Users/sharadc/Documents/uic/extra/repos/archive/asl_alphabet_train/asl_alphabet_train/A/A1.jpg")
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = hands.process(imgRGB)
print(results.multi_hand_landmarks)

hands2DList = [];
# for x,y,z in results.multi_hand_landmarks:
#     print("dasdas");
# lst = [[x,y,z] for x,y,z in results.multi_hand_landmarks]
if results.multi_hand_landmarks:
    for handLms in results.multi_hand_landmarks:
        lst = [];
        for id, lm in enumerate(handLms.landmark):
            #print(id,lm)
            h, w, c = img.shape
            cx, cy = int(lm.x *w), int(lm.y*h)
            lst.append([cx, cy]);
            #if id ==0:
            cv2.circle(img, (cx,cy), 7, (255,0,255), cv2.FILLED)
        hands2DList.append(lst);
        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

cTime = time.time()
fps = 1/(cTime-pTime)
pTime = cTime
hands2DListnp = np.array(hands2DList);
centroids_for_each_hand = [];
opposite_points_for_each_hand=[];
XS = 200;
YS = 150;
create_rectangle_around_hand(hands2DList, img,XS,YS);

cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

cv2.imshow("Image", img)
cv2.waitKey()
cv2.destroyAllWindows()


