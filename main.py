import cv2
import dlib

# Load the pre-trained facial landmark predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

# Load the image
image_path = "FACES_database/004_o_m_h_a.jpg"
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use the dlib detector to find faces in the image
detector = dlib.get_frontal_face_detector()
faces = detector(gray)

# Loop over each face
for face in faces:
    # Get the facial landmarks
    landmarks = predictor(gray, face)

    # Example: Calculate distances between specific facial landmarks (customize based on FACS)
    # For example, distance between eyebrows and mouth
    eyebrow_to_mouth_distance = landmarks.part(66).y - landmarks.part(62).y
    eye_blink = landmarks.part(47).y - landmarks.part(43).y
    smile_intensity = landmarks.part(54).x - landmarks.part(48).x
    frown_intensity = landmarks.part(59).y - landmarks.part(55).y

    # Example: Determine emotion based on distances and FACS indicators
    emotion = "Neutral"
    if eyebrow_to_mouth_distance > 2 and eye_blink > 5:
        emotion = "Happy"
    elif eyebrow_to_mouth_distance < -2 and frown_intensity > 5:
        emotion = "Sad"
    elif smile_intensity > 10:
        emotion = "Smiling"

    # Return emotion
    cv2.putText(image, "Emotion: " + emotion, (10, image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 255, 0), 8, cv2.LINE_AA)

# Display the result
cv2.namedWindow("Emotion Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Emotion Recognition", 800, 600)
cv2.imshow("Emotion Recognition", image)
cv2.waitKey(0)
cv2.destroyAllWindows()