import cv2
import dlib

# Load the pre-trained facial landmark predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)


# Function will determine the emotion
def emotion_determiner(shot, gray_scaled):
    # Loop over each face if there's multiple
    for face in shot:
        # Get the facial landmarks
        landmarks = predictor(gray_scaled, face)

        # Calculate facial distances
        eyebrow_to_mouth_distance = landmarks.part(66).y - landmarks.part(62).y
        eye_blink = landmarks.part(47).y - landmarks.part(43).y
        smile_intensity = landmarks.part(54).x - landmarks.part(48).x
        frown_intensity = landmarks.part(59).y - landmarks.part(55).y

        # Determine emotion based on distances
        emotion = "Neutral"
        if eyebrow_to_mouth_distance > 2 and eye_blink > 5:
            emotion = "Happy"
        elif eyebrow_to_mouth_distance < -2 and frown_intensity > 5:
            emotion = "Sad"
        elif smile_intensity > 10:
            emotion = "Smiling"

        # Return emotion
        cv2.putText(image, "Emotion: " + emotion, (10, image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 255, 0),
                    8, cv2.LINE_AA)
        # Display the result
        cv2.namedWindow("Emotion Recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Emotion Recognition", 800, 600)
        cv2.imshow("Emotion Recognition", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Ask for image or camera
choice = input("Image or camera? (y/n)")

# Set to either image or camera
if choice == "y":
    # Load the image
    image_path = "FACES_database/004_o_m_h_a.jpg"
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)
    emotion_determiner(faces, gray)
elif choice == "n":
    print()


