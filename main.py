import cv2
import dlib
import os

# Load the pre-trained facial landmark predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

# Set up FACES database
facesDB = "./FACES_database/"

# add facial framing. for example, distance can change a face's ratios. need 2 fix this

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
        upper_lid_raiser = abs(landmarks.part(19).y - landmarks.part(41).y)/100

        print(f"Cheek Raiser: {cheek_raiser}")
        print(f"Lip Corner Puller: {lip_corner_puller}")
        print(f"Outer Brow Distance: {outer_brow_distance}")
        print(f"Inner Brow Distance: {inner_brow_distance}")
        print(f"Mid Brow Distance: {mid_brow_distance}")
        print(f"Lid Tightness: {lid_tightness}")
        print(f"Lip Corner Depressor: {lip_corner_depressor}")
        print(f"Lip Tightness: {lip_tightness}")
        print(f"Jaw Drop: {jaw_drop}")
        print(f"Nose Wrinkle: {nose_wrinkle}")
        print(f"Upper Lid Raiser: {upper_lid_raiser}")

        # Determine emotion based on indicators
        emotion = "Neutral"
        if cheek_raiser > 6.6 and (lip_corner_puller > 0.06 and lip_corner_puller < 0.16):
            emotion = "Happy"
            if inner_brow_distance < 2.5 and lip_corner_depressor > 6.8 and mid_brow_distance < 2.95:
                emotion = "Sad"
        elif mid_brow_distance < 1.5 and lid_tightness < 1.3 and lip_tightness < 1.3 and upper_lid_raiser < 2.2:
            emotion = "Anger"
        elif inner_brow_distance < 2.5 and lip_corner_depressor > 6.8 and mid_brow_distance < 2.95:
            emotion = "Sad"
        elif nose_wrinkle > 3.1 and lip_corner_depressor < 6.35 and lip_tightness > 1.3:
            emotion = "Disgust"
        elif (inner_brow_distance > 2.24 or outer_brow_distance > 1.2) and (mid_brow_distance < 3.2 and lid_tightness < 1.4) and jaw_drop > 4.4 and upper_lid_raiser > 3.7:
            emotion = "Fear"


        print(f"Predicted Emotion: {emotion}")

        if show:
            # Returns and displays emotion
            cv2.putText(image, "Emotion: " + emotion, (10, image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 255, 0),8, cv2.LINE_AA)
            cv2.namedWindow("Emotion Recognition", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Emotion Recognition", 800, 600)
            cv2.imshow("Emotion Recognition", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return emotion


# Ask for image or camera
choice = input("Image or camera? (y/n)")

# Set to either image or camera
if choice == "y":
    # Load the image
    # h = happy a = angry f = fearful s = sad d = disgusted n = neutral
    image_path = "./FACES_database/004_o_m_d_a.jpg"
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)
    emotion_determiner(faces, gray, False)
elif choice == "g":
    correct = 0
    total = 0
    happy = 0
    sad = 0
    anger = 0
    fear = 0
    neutral = 0
    disgust = 0
    for file in os.listdir(facesDB):
        total += 1
        image_path = os.path.join(facesDB, file)  # Construct full file path
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray)
        predictedEmotion = emotion_determiner(faces, gray, False)
        actualEmotion = file[8]
        if actualEmotion == "h":
            actualEmotion = "Happy"
        elif actualEmotion == "a":
            actualEmotion = "Anger"
        elif actualEmotion == "n":
            actualEmotion = "Neutral"
        elif actualEmotion == "s":
            actualEmotion = "Sad"
        elif actualEmotion == "f":
            actualEmotion = "Fear"
        elif actualEmotion == "d":
            actualEmotion = "Disgust"
        print(f"Actual Emotion: {actualEmotion}")
        if actualEmotion == predictedEmotion:
            correct += 1
    print(f"Correct: {correct}")
    print(f"Total: {total}")
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
        emotion_determiner(faces, gray, True)

        # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('u'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()