# USAGE
# python imageid.py --dataset testimages --embeddings output/embeddings.pickle 
# --detector face_detection_model --embedding-model nn4.small2.v1.t7 --ineligible output/ineligible.txt
# -- s3bucket bucket key

from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import boto3
from PIL import Image
import io
import requests
import urllib
import openpyxl
from datetime import datetime

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=False,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--embeddings", required=False,
	help="path to output serialized db of facial embeddings")
ap.add_argument("-d", "--detector", required=False, default="face_detection_model",
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=False, default="nn4.small2.v1.t7",
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--ineligible", required=False,
	help="path to low confidence images list")
ap.add_argument("-s3b", "--s3bucket", nargs=2, required=False,
	help="Amazon S3 bucket and key")
ap.add_argument("-x", "--xlsx", required=False,
	help="Xlsx file of URL path")
ap.add_argument("-r", "--range", required=True,
	help="Range of vectors to be generated")

args = vars(ap.parse_args())

print(datetime.now())
s3 = boto3.resource('s3')

# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image

if(args["xlsx"]):
	path = args["xlsx"]
	wb = openpyxl.load_workbook(path)
	sheet = wb.active

def import_from_workbook():
	# Give the location of the file
	
	for i in range (1, int(args["range"])):
		imagePaths.append(sheet["A"+str(i)].value)
	
	wb.close()

def write_to_workbook(vector, i):

	sheet["B"+str(i+1)] = vector
	

def image_from_s3(bucket, key):

    bucket = s3.Bucket(bucket)
    image = bucket.Object(key)
    img_data = image.get().get('Body').read()

    return Image.open(io.BytesIO(img_data))

# load the face detector from disk
print("Loading Caffe based face detector to localize faces in an image")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the face embedding model from disk
print("Loading Openface imlementation of Facenet model")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# grab the paths to the input images in our dataset
print("Load image dataset..")

if args["xlsx"]:
	imagePaths = []
	import_from_workbook()
elif args["s3bucket"]:
	imagePaths = image_from_s3(args["s3bucket"][0], args["s3bucket"][1])
	print(args["s3bucket"][0])
else:
	imagePaths = list(paths.list_images(args["dataset"]))


# initialize our lists of extracted facial embeddings and
# corresponding people names
embedding_dict = {}
ineligible = []

# initialize the total number of faces processed
total = 0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	# name = imagePath.split(os.path.sep)[-2]

	# load the image, resize it to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	if(args["xlsx"]):
		image = url_to_image(imagePath)
	else:
		image = cv2.imread(imagePath) 
	
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# ensure at least one face was found
	if len(detections) > 0:
		# we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# ensure that the detection with the 50% probabilty thus helping filter out weak detections
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI and grab the ROI dimensions
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				print("Dimension error for "+ imagePath)
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# add the name of the person + corresponding face
			# embedding to their respective lists
			flatvector = vec.flatten()
			
			embedding_dict[imagePath] = (flatvector)
			
			
			if(args["xlsx"]):
				write_to_workbook(str(flatvector), total)
			total += 1

		else:
			ineligible.append(imagePath.split())
			print("Confidence too low for: " + imagePath)

if(args["xlsx"]):		
	wb.save(args["xlsx"])

# f = open("output/embeddings.txt", "w")
# for k in embedding_dict.keys():
#     f.write("{}:\n{}\n\n".format(k, embedding_dict[k]))


with open(args["ineligible"], mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(str(line) for line in ineligible))

if(args["embeddings"]):
	f = open(args["embeddings"], "wb")
	f.write(pickle.dumps(embedding_dict))
	f.close()
	print(datetime.now())