import cv2

img = cv2.imread('result.jpg')
cv2.imshow('RGB IMAGE', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
