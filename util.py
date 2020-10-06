from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import imutils
import cv2

confidence_bound = 0.5

def mask_prediction(frame, face_net, mask_net):

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

	face_net.setInput(blob)
	detections = face_net.forward()

	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]

		if confidence > confidence_bound:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(start_x, start_y, end_x, end_y) = box.astype("int")

			(start_x, start_y) = (max(0, start_x), max(0, start_y))
			(end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))

			face = frame[start_y:end_y, start_x:end_x]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((start_x, start_y, end_x, end_y))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = mask_net.predict(faces, batch_size=32)

	return (locs, preds)