import cv2
import dlib
import os

# Load the pre-trained facial landmark predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

# Set up FACES database
facesDB = "./FACES_database/"

# Function will determine the emotion
def emotion_determiner(shot, gray_scaled, show):
    # Loop over each face if there's multiple
    for face in shot:
        # Get the facial landmarks
        landmarks = predictor(gray_scaled, face)

        # Indicators from the Facial Action Coding System
        outer_brow_distance = abs(landmarks.part(17).y - landmarks.part(36).y)/100
        inner_brow_distance = abs(landmarks.part(20).y - landmarks.part(38).y)/100
        mid_brow_distance = abs(landmarks.part(19).y - landmarks.part(37).y)/100
        lid_tightness = abs(landmarks.part(41).y - landmarks.part(37).y)/100
        lip_corner_depressor = abs(landmarks.part(48).x - landmarks.part(54).x)/100
        lip_corner_puller = abs(landmarks.part(54).y - landmarks.part(48).y)/100
        lip_tightness = abs(landmarks.part(51).y - landmarks.part(57).y)/100
        jaw_drop = abs(landmarks.part(57).y - landmarks.part(8).y)/100
        nose_wrinkle = abs(landmarks.part(31).x - landmarks.part(35).x)/100
        cheek_raiser = abs(landmarks.part(1).y - landmarks.part(4).y)/100
        print(str(cheek_raiser)+":cheekraiser")
        print(str(lip_corner_puller)+":lipcornerpuller")
        print(str(outer_brow_distance) + ":outerbrowdistance")
        print(str(inner_brow_distance) + ":innerbrowdistance")
        print(str(mid_brow_distance) + ":midbrowdistance")
        print(str(cheek_raiser) + ":cheekraiser")
        print(str(lip_corner_puller) + ":lipcornerpuller")
        print(str(lip_corner_depressor)+":lipcornerdepressor")
        # Determine emotion based on indicators
        emotion = "Neutral"
        if cheek_raiser > 7.1 and lip_corner_puller > 0.1:
            emotion = "Happy"
        elif inner_brow_distance < 1.5 < lip_corner_depressor and mid_brow_distance < 1.8:
            emotion = "Sad"
        elif inner_brow_distance > 1.0 or (outer_brow_distance > 1.0 and jaw_drop > 2.0):
            emotion = "Surprise"
        elif (inner_brow_distance > 1.0 or outer_brow_distance > 1.0) and (
                mid_brow_distance < 1.8 and lid_tightness < 1.8) and jaw_drop > 2.0:
            emotion = "Fear"
        elif mid_brow_distance < 1.8 and lid_tightness < 2.0 and lip_tightness < 2.0:
            emotion = "Anger"
        elif nose_wrinkle > 1.1 and lip_corner_depressor > 1.1:
            emotion = "Disgust"

        if show:
            # Returns and displays emotion
            cv2.putText(image, "Emotion: " + emotion, (10, image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 255, 0),8, cv2.LINE_AA)
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
    # h = happy a = angry f = fearful s = sad d = disgusted n = neutral
    image_path = "./FACES_database/066_y_m_d_a.jpg"
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)
    emotion_determiner(faces, gray, True)
elif choice == "g":
    for file in os.listdir(facesDB):
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray)
        emotion_determiner(faces, gray, False)
elif choice == "n":
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, image = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use the dlib detector to find faces in the frame
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray)

        # Call the emotion determiner function
        emotion_determiner(faces, gray)

        # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('u'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

