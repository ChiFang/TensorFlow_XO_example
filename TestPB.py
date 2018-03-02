import cv2

ModelFile = "OX_Predict_frozen.pb"

print("Start...")
net = cv2.dnn.readNetFromTensorflow(ModelFile)
print("Load \'" + ModelFile + "\' Pass~~")
# frame = cv2.imread("ci1.bmp")
frame = cv2.imread("ci1.bmp", cv2.IMREAD_GRAYSCALE)
inScaleFactor = 1 / 255
print("Read image Pass~~")
blob = cv2.dnn.blobFromImage(frame, inScaleFactor, (116, 116))
print("blobFromImage Pass~~")
net.setInput(blob)
print("setInput Pass~~")
detections = net.forward()
print(detections)

