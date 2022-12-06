import cv2
import mediapipe as mp
import time
import numpy as np;
import InferOneImage as inferer;

def create_rectangle_around_hand(hands2DList, img, XS, YS):
    hands2DListnp = np.array(hands2DList);
    #print(hands2DListnp.shape);
    centroids_for_each_hand = [];
    opposite_points_for_each_hand = [];
    XS = 350;
    YS = 350;
    for i in range(0, hands2DListnp.shape[0]):
        handLandmarks = hands2DListnp[i];
        sum_row = np.sum(handLandmarks[:, 0]);
        sum_column = np.sum(handLandmarks[:, 1]);
        centroid = [sum_row / handLandmarks.shape[0], sum_column / handLandmarks.shape[0]];
        opposite_points = [[centroid[0] - XS, centroid[1] - YS], [centroid[0] + XS, centroid[1] + YS]];
        opposite_points_right_top_left_bottom = [[centroid[0] - XS, centroid[1] + YS], [centroid[0] + XS, centroid[1] - YS]];
        opposite_points_np = np.array(opposite_points);
        opposite_points_np_int = opposite_points_np.astype("int");
        opposite_points_right_top_left_bottom_np = np.array(opposite_points_right_top_left_bottom);
        opposite_points_right_top_left_bottom_np_int = opposite_points_right_top_left_bottom_np.astype(int);
        #print((opposite_points_np_int[0, 0], opposite_points_np_int[0, 1]));
        #print((opposite_points_np_int[1, 0], opposite_points_np_int[1, 1]));
        c1 = (opposite_points_np_int[0, 0], opposite_points_np_int[0, 1]);
        c2 = (opposite_points_np_int[1, 0], opposite_points_np_int[1, 1]);
        cv2.rectangle(img, c1,
                      c2, (0, 255, 0), 3);
        #centroids_for_each_hand.append(centroid);
        #opposite_points_for_each_hand.append(opposite_points);
        c3= (opposite_points_right_top_left_bottom_np_int[0, 0], opposite_points_right_top_left_bottom_np_int[0, 1]);
        c4= (opposite_points_right_top_left_bottom_np_int[1, 0], opposite_points_right_top_left_bottom_np_int[1, 1]);
        recX = [opposite_points_np_int[0, 0], opposite_points_np_int[1, 0], opposite_points_right_top_left_bottom_np_int[0, 0], opposite_points_right_top_left_bottom_np_int[1, 0]];
        recY = [opposite_points_np_int[1, 0], opposite_points_np_int[1, 1], opposite_points_right_top_left_bottom_np_int[1, 0], opposite_points_right_top_left_bottom_np_int[1, 1]];
        top_left_x = min(recX)
        top_left_y = min(recY)
        bot_right_x = max(recX)
        bot_right_y = max(recY)
        op=img[top_left_y:bot_right_y, top_left_x:bot_right_x];
        ls=[op];
        y=inferer.runInferenceVGG16(ls);
        return op,y;



def captureVideoAndAnnotateHand(capture_duration=40):
    global success, results, id, w, c
    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    pTime = 0
    cTime = 0
    start_time = time.time()
    imgs = [];
    op = np.zeros((500,500));
    y = "#"
    while int(time.time() - start_time) < capture_duration :
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        hands2DList = [];
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                lst = [];
                for id, lm in enumerate(handLms.landmark):
                    # print(id,lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lst.append([cx, cy]);
                    # if id ==0:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
                    hands2DList.append(lst);
                #mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            XS = 300;
            YS = 250;
            #print(hands2DList);
            op,y = create_rectangle_around_hand(hands2DList, img, XS, YS);

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        #cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.putText(op, str(y),  (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3);
        cv2.imshow("Image", op)
        imgs.append(op);
        cv2.waitKey(1)

    return imgs,fps;


captureVideoAndAnnotateHand(capture_duration=60);