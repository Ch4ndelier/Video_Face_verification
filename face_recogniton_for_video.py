import face_recognition
import cv2
import os

# Often instead of just checking if two faces match or not (True or False), it's helpful to see how similar they are.
# You can do that by using the face_distance function.

# The model was trained in a way that faces with a distance of 0.6 or less should be a match. But if you want to
# be more strict, you can look for a smaller face distance. For example, using a 0.55 cutoff would reduce false
# positive matches at the risk of more false negatives.

# Note: This isn't exactly the same as a "percent match". The scale isn't linear. But you can assume that images with a
# smaller distance are more similar to each other than ones with a larger distance.

v_dir = "video/"
if not os.path.exists('temp/'):
    os.mkdir('temp')
count = 0

for eachVid in os.listdir(v_dir):
    vPath = os.path.join(v_dir, eachVid)
    temp_path = os.path.join('temp/',eachVid)
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)

    vidcap = cv2.VideoCapture(vPath)
    success, image = vidcap.read()
    timeF = 30
    c = 1
    while success:
        success, image = vidcap.read()
        if c % timeF == 0:
            cv2.imwrite(os.path.join(temp_path, "{}.png".format(count)), image)     # save frame as JPEG file
            count += 1
            if count%5 == 0:
                break
        c = c + 1
    vidcap.release()
        #print('Read a new frame: ', success)

#image to test
image_to_test = face_recognition.load_image_file("person.jpg")
image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]

for eachjudge in os.listdir("temp"):
    known_encodings = []
    for eachpic in os.listdir("temp/" + eachjudge):
        pic_path = os.path.join("temp", eachjudge, eachpic)
        image = face_recognition.load_image_file(pic_path)
        image_encoding = face_recognition.face_encodings(image)[0]
        known_encodings.append(image_encoding)
    face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)
    print(face_distances)
    count = 0
    for distance in face_distances:
        if distance > 0.55:
            count += 1
    if count >= 4:
        print(eachjudge, "False")
    else:
        print(eachjudge, "True")

