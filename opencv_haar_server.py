import cv2
from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def face_detection():
  filename = request.args.get('filename', 'conf2.jpg')
  scale_factor = float(request.args.get('scale_factor', '1.1'))
  min_neighbors = int(request.args.get('min_neighbors', '3'))
  return detect_faces_and_save(filename, scale_factor, min_neighbors)

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def detect_faces_and_save(filename, scale_factor, min_neighbors):
  # load cascade classifier training file for haarcascade
  haar_face_cascade = cv2.CascadeClassifier('/Users/max/work/sumatosoft/libs/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')

  # load iamge
  test_img = cv2.imread(filename)

  # convert the test image to gray image as opencv face detector expects gray images
  gray_img = convertToRGB(test_img)

  faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=scale_factor, minNeighbors=min_neighbors)

  for (x, y, w, h) in faces:
      cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

  sentence = filename.split('.')
  output_filename = '_'.join(sentence[0:-1]) + '_result.' + sentence[-1]
  cv2.imwrite(output_filename, test_img)

  return output_filename

if __name__ == "__main__":
    app.run()
