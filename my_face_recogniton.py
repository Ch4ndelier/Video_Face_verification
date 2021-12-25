import face_recognition

# Often instead of just checking if two faces match or not (True or False), it's helpful to see how similar they are.
# You can do that by using the face_distance function.

# The model was trained in a way that faces with a distance of 0.6 or less should be a match. But if you want to
# be more strict, you can look for a smaller face distance. For example, using a 0.55 cutoff would reduce false
# positive matches at the risk of more false negatives.

# Note: This isn't exactly the same as a "percent match". The scale isn't linear. But you can assume that images with a
# smaller distance are more similar to each other than ones with a larger distance.

# Load some images to compare against
image_0 = face_recognition.load_image_file("data/target_0.png")
image_1 = face_recognition.load_image_file("data/target_1.png")
image_2 = face_recognition.load_image_file("data/other_0.png")
image_3 = face_recognition.load_image_file("data/other_1.png")

# Get the face encodings for the known images
image_0_encoding = face_recognition.face_encodings(image_0)[0]
image_1_encoding = face_recognition.face_encodings(image_1)[0]
image_2_encoding = face_recognition.face_encodings(image_2)[0]
image_3_encoding = face_recognition.face_encodings(image_3)[0]

known_encodings = [
    image_0_encoding,
    image_1_encoding,
    image_2_encoding,
    image_3_encoding
]

# Load a test image and get encondings for it
image_to_test = face_recognition.load_image_file("smile.jpg")
image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]
print(image_to_test_encoding.shape)
# See how far apart the test image is from the known faces
face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)
#face_distances = face_recognition.compare_faces(known_encodings, image_to_test_encoding)
#print(face_distances)
for i, face_distance in enumerate(face_distances):
    print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
    print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
    print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(face_distance < 0.5))
    print()
