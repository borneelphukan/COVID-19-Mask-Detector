
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

dataset_path = "./dataset"
model_path = "borneel_mask_detector.model"

init_lr = 1e-4
epochs = 20
batch_size = 32

imagePaths = list(paths.list_images(dataset_path))
data = []
labels = []

for image in imagePaths:
	# extract the class label from the filename
	label = image.split(os.path.sep)[-2]

	image = load_img(image, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	data.append(image)
	labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Data splitting
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=0)

image_data_generator = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network, classifying layer off
mobile_net = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

custom_net = mobile_net.output
custom_net = AveragePooling2D(pool_size=(7, 7))(custom_net)
custom_net = Flatten(name="flatten")(custom_net)
custom_net = Dense(128, activation="relu")(custom_net)
custom_net = Dropout(0.5)(custom_net)
custom_net = Dense(2, activation="softmax")(custom_net)

# sequencing the network
model = Model(inputs=mobile_net.input, outputs=custom_net)

# Only train the custom section of the network
for layer in mobile_net.layers:
	layer.trainable = False

# compile our model
model.compile(loss="binary_crossentropy", optimizer=Adam(lr=init_lr, decay=init_lr / epochs), metrics=["accuracy"])

# train the head of the network
train_custom = model.fit(
	image_data_generator.flow(train_x, train_y, batch_size=batch_size),
	steps_per_epoch=len(train_x) // batch_size,
	validation_data=(test_x, test_y),
	validation_steps=len(test_x) // batch_size,
	epochs=epochs)

# make predictions on the testing set
prediction = model.predict(test_x, batch_size=batch_size)

# index of the label with largest probability for each image
prediction = np.argmax(prediction, axis=1)

# classification report
print(classification_report(test_y.argmax(axis=1), prediction,
	target_names=lb.classes_))

# save the model to disk
model.save(model_path, save_format="h5")

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), train_custom.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), train_custom.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), train_custom.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), train_custom.history["val_accuracy"], label="val_acc")
plt.title("Training Loss & Accuracy Graph")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig("plot.png")