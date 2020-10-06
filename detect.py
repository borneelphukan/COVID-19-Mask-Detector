from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
from util import mask_prediction
import imutils
import cv2
import os

face_detector = "./face_detector"
model_path = "borneel_mask_detector.model"

# load pretrained face detector
prototxtPath = os.path.sep.join([face_detector, "deploy.prototxt"])
weightsPath = os.path.sep.join([face_detector,"res10_300x300_ssd_iter_140000.caffemodel"])
face_net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load mask detector model
mask_net = load_model(model_path)

video = VideoStream(0).start()

while True:
	frame = video.read()
	frame = imutils.resize(frame, width=800)

	(locs, preds) = mask_prediction(frame, face_net, mask_net)

	for (box, pred) in zip(locs, preds):
		(start_x, start_y, end_x, end_y) = box
		(mask, withoutMask) = pred

		if mask > withoutMask:
			label = "Mask"
			color = (0, 255, 0)
		else:
			label = "No mask"
			color = (0, 0, 255)

		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == 27:
		break

cv2.destroyAllWindows()
video.stop()
