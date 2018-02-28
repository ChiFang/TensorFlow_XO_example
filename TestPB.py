import cv2


print("1...")
net = cv2.dnn.readNetFromTensorflow("OX_Predict_frozen.pb")
print("2...")
frame = cv2.imread("ci1.bmp")
inScaleFactor = 1 / 255
print("3...")
blob = cv.dnn.blobFromImage(frame, inScaleFactor, (116, 116))
print("4...")
net.setInput(blob)
print("5...")
detections = net.forward()

print(detections)

