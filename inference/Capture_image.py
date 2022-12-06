import cv2;
import numpy as np;

im = cv2.imread("../opencv_frame_1.png");
im = im.astype(float);
cv2.imshow("hither",im.astype(np.uint8));
cv2.waitKey()
cv2.destroyAllWindows()