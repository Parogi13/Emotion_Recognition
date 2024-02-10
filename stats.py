import cv2
import dlib
import os

# Load the pre-trained facial landmark predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

# Set up FACES database
facesDB = "./FACES_database/"

# Initialize variables for counting faces detected for each emotion
happy_count = 0
anger_count = 0
sad_count = 0
fear_count = 0
disgust_count = 0

# Initialize lists to accumulate indicators for each emotion
happy_indicators = [0, 0]  # cheek_raiser, lip_corner_puller
anger_indicators = [0, 0, 0, 0]  # lid_tightness, lip_tightness, mid_brow_distance, upper_lid_raiser
sad_indicators = [0, 0, 0]  # mid_brow_distance, inner_brow_distance, outer_brow_distance
fear_indicators = [0, 0, 0, 0, 0,
                   0]  # jaw_drop, inner_brow_distance, mid_brow_distance, outer_brow_distance, lid_tightness, upper_lid_raiser
disgust_indicators = [0, 0, 0]  # lid_tightness, lip_tightness, jaw_drop


# Function will determine the emotion
def emotion_determiner(shot, gray_scaled, actual_emotion):
    global happy_count, anger_count, sad_count, fear_count, disgust_count
    global happy_indicators, anger_indicators, sad_indicators, fear_indicators, disgust_indicators

    # Loop over each face if there's multiple
    for face in shot:
        # Get the facial landmarks
        landmarks = predictor(gray_scaled, face)

        # Indicators from the Facial Action Coding System
        outer_brow_distance = abs(landmarks.part(17).y - landmarks.part(36).y) / 100
        inner_brow_distance = abs(landmarks.part(20).y - landmarks.part(38).y) / 100
        mid_brow_distance = abs(landmarks.part(19).y - landmarks.part(37).y) / 100
        lid_tightness = abs(landmarks.part(41).y - landmarks.part(37).y) / 100
        lip_corner_depressor = abs(landmarks.part(48).x - landmarks.part(54).x) / 100
        lip_corner_puller = abs(landmarks.part(54).y - landmarks.part(48).y) / 100
        lip_tightness = abs(landmarks.part(51).y - landmarks.part(57).y) / 100
        jaw_drop = abs(landmarks.part(57).y - landmarks.part(8).y) / 100
        nose_wrinkle = abs(landmarks.part(31).x - landmarks.part(35).x) / 100
        cheek_raiser = abs(landmarks.part(1).y - landmarks.part(4).y) / 100
        upper_lid_raiser = abs(landmarks.part(19).y - landmarks.part(41).y) / 100

        # Accumulate indicators based on the detected emotion
        if actual_emotion == "Happy":
            happy_indicators[0] += cheek_raiser
            happy_indicators[1] += lip_corner_puller
            happy_count += 1
        elif actual_emotion == "Sad":
            sad_indicators[0] += mid_brow_distance
            sad_indicators[1] += inner_brow_distance
            sad_indicators[2] += outer_brow_distance
            sad_count += 1
        elif actual_emotion == "Disgust":
            disgust_indicators[0] += lid_tightness
            disgust_indicators[1] += lip_tightness
            disgust_indicators[2] += jaw_drop
            disgust_count += 1
        elif actual_emotion == "Fear":
            fear_indicators[0] += jaw_drop
            fear_indicators[1] += inner_brow_distance
            fear_indicators[2] += mid_brow_distance
            fear_indicators[3] += outer_brow_distance
            fear_indicators[4] += lid_tightness
            fear_indicators[5] += upper_lid_raiser
            fear_count += 1
        elif actual_emotion == "Anger":
            anger_indicators[0] += lid_tightness
            anger_indicators[1] += lip_tightness
            anger_indicators[2] += mid_brow_distance
            anger_indicators[3] += upper_lid_raiser
            anger_count += 1

    # Print the average indicators for each emotion
    if happy_count > 0:
        print(
            f"Happy - cheek raiser: {happy_indicators[0] / happy_count} lip corner puller: {happy_indicators[1] / happy_count}")
    else:
        print("No happy faces detected")
    if sad_count > 0:
        print(
            f"Sad - mid brow distance: {sad_indicators[0] / sad_count} inner brow distance: {sad_indicators[1] / sad_count} outer brow distance: {sad_indicators[2] / sad_count}")
    else:
        print("No sad faces detected")
    if disgust_count > 0:
        print(
            f"Disgust - lid tightness: {disgust_indicators[0] / disgust_count} lip tightness: {disgust_indicators[1] / disgust_count} jaw drop: {disgust_indicators[2] / disgust_count}")
    else:
        print("No disgust faces detected")
    if fear_count > 0:
        print(
            f"Fear - jaw drop: {fear_indicators[0] / fear_count} inner brow distance: {fear_indicators[1] / fear_count} mid brow distance: {fear_indicators[2] / fear_count} outer brow distance: {fear_indicators[3] / fear_count} lid tightness: {fear_indicators[4] / fear_count} upper lid raiser: {fear_indicators[5] / fear_count}")
    else:
        print("No fear faces detected")
    if anger_count > 0:
        print(
            f"Anger - lid tightness: {anger_indicators[0] / anger_count} lip tightness: {anger_indicators[1] / anger_count} mid brow distance: {anger_indicators[2] / anger_count} upper lid raiser: {anger_indicators[3] / anger_count}")
    else:
        print("No anger faces detected")


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
    emotion_determiner(faces, gray, "Disgust")
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
        actualEmotion = file[8]
        if actualEmotion == "h":
            actualEmotion = "Happy"
            happy += 1
        elif actualEmotion == "a":
            actualEmotion = "Anger"
            anger += 1
        elif actualEmotion == "n":
            actualEmotion = "Neutral"
            neutral += 1
        elif actualEmotion == "s":
            actualEmotion = "Sad"
            sad += 1
        elif actualEmotion == "f":
            actualEmotion = "Fear"
            fear += 1
        elif actualEmotion == "d":
            actualEmotion = "Disgust"
            disgust += 1
        print(f"Actual Emotion: {actualEmotion}")
        emotion_determiner(faces, gray, actualEmotion)
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

        # Break the loop when the 'u' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('u'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
