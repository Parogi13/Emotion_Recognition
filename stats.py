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
fear_indicators = [0, 0, 0, 0, 0,0]  # jaw_drop, inner_brow_distance, mid_brow_distance, outer_brow_distance, lid_tightness, upper_lid_raiser
disgust_indicators = [0, 0, 0]  # lid_tightness, lip_tightness, jaw_drop

#Happy
happy_cheek_raiser = []
happy_lip_corner_puller = []

#Sad
sad_mid_brow_distance = []
sad_inner_brow_distance = []
sad_lip_corner_depressor = []

#Disgust
disgust_nose_wrinkle = []
disgust_lip_tightness= []
disgust_lip_corner_depressor = []

#Fear
fear_jaw_drop= []
fear_inner_brow_distance= []
fear_mid_brow_distance= []
fear_outer_brow_distance= []
fear_lid_tightness= []
fear_upper_lid_raiser= []

#Anger
anger_lid_tightness= []
anger_lip_tightness= []
anger_mid_brow_distance= []
anger_upper_lid_raiser= []

# Function will determine the emotion
def emotion_determiner(shot, gray_scaled, actual_emotion):
    global happy_count, anger_count, sad_count, fear_count, disgust_count
    global happy_indicators, anger_indicators, sad_indicators, fear_indicators, disgust_indicators

    # Loop over each face if there's multiple
    for face in shot:
        # Get the facial landmarks
        landmarks = predictor(gray_scaled, face)

        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        # Indicators from the Facial Action Coding System
        outer_brow_distance = abs(landmarks.part(17).y - landmarks.part(36).y) / (h)
        inner_brow_distance = abs(landmarks.part(21).y - landmarks.part(39).y) / (h)
        mid_brow_distance = abs(landmarks.part(19).y - landmarks.part(37).y) / (h)
        lid_tightness = abs(landmarks.part(41).y - landmarks.part(37).y) / (h)
        lip_corner_depressor = abs(landmarks.part(51).y - landmarks.part(48).y) / (h)
        lip_corner_puller = abs(landmarks.part(51).x - landmarks.part(48).x) / (w)
        lip_tightness = abs(landmarks.part(54).x - landmarks.part(48).x) / (w)
        jaw_drop = abs(landmarks.part(33).y - landmarks.part(8).y) / (h)
        nose_wrinkle = abs(landmarks.part(31).x - landmarks.part(35).x) / (w)
        cheek_raiser = abs(landmarks.part(36).y - landmarks.part(2).y) / (h)
        upper_lid_raiser = abs(landmarks.part(19).y - landmarks.part(41).y) / (h)

        # Accumulate indicators based on the detected emotion
        if actual_emotion == "Happy":
            happy_indicators[0] += cheek_raiser
            happy_indicators[1] += lip_corner_puller
            happy_cheek_raiser.append(cheek_raiser)
            happy_lip_corner_puller.append(lip_corner_puller)
            happy_count += 1
        elif actual_emotion == "Sad":
            sad_indicators[0] += mid_brow_distance
            sad_indicators[1] += inner_brow_distance
            sad_indicators[2] += lip_corner_depressor
            sad_mid_brow_distance.append(mid_brow_distance)
            sad_inner_brow_distance.append(inner_brow_distance)
            sad_lip_corner_depressor.append(lip_corner_depressor)
            sad_count += 1
        elif actual_emotion == "Disgust":
            disgust_indicators[0] += nose_wrinkle
            disgust_indicators[1] += lip_tightness
            disgust_indicators[2] += lip_corner_depressor
            disgust_nose_wrinkle.append(nose_wrinkle)
            disgust_lip_tightness.append(lip_tightness)
            disgust_lip_corner_depressor.append(lip_corner_depressor)
            disgust_count += 1
        elif actual_emotion == "Fear":
            fear_indicators[0] += jaw_drop
            fear_indicators[1] += inner_brow_distance
            fear_indicators[2] += mid_brow_distance
            fear_indicators[3] += outer_brow_distance
            fear_indicators[4] += lid_tightness
            fear_indicators[5] += upper_lid_raiser
            fear_jaw_drop.append(jaw_drop)
            fear_inner_brow_distance.append(inner_brow_distance)
            fear_outer_brow_distance.append(outer_brow_distance)
            fear_mid_brow_distance.append(mid_brow_distance)
            fear_upper_lid_raiser.append(upper_lid_raiser)
            fear_lid_tightness.append(lid_tightness)
            fear_count += 1
        elif actual_emotion == "Anger":
            anger_indicators[0] += lid_tightness
            anger_indicators[1] += lip_tightness
            anger_indicators[2] += mid_brow_distance
            anger_indicators[3] += upper_lid_raiser
            anger_lid_tightness.append(lid_tightness)
            anger_lip_tightness.append(lip_tightness)
            anger_mid_brow_distance.append(mid_brow_distance)
            anger_upper_lid_raiser.append(upper_lid_raiser)
            anger_count += 1

    # Print the average indicators for each emotion
    if happy_count > 0:
        print(f"Happy - cheek raiser: {happy_indicators[0] / happy_count} lip corner puller: {happy_indicators[1] / happy_count}")
    else:
        print("No happy faces detected")
    if sad_count > 0:
        print(f"Sad - mid brow distance: {sad_indicators[0] / sad_count} inner brow distance: {sad_indicators[1] / sad_count} lip corner depressor: {sad_indicators[2] / sad_count}")
    else:
        print("No sad faces detected")
    if disgust_count > 0:
        print(f"Disgust - nose wrinkle: {disgust_indicators[0] / disgust_count} lip tightness: {disgust_indicators[1] / disgust_count} lip corner depressor: {disgust_indicators[2] / disgust_count}")
    else:
        print("No disgust faces detected")
    if fear_count > 0:
        print(f"Fear - jaw drop: {fear_indicators[0] / fear_count} inner brow distance: {fear_indicators[1] / fear_count} mid brow distance: {fear_indicators[2] / fear_count} outer brow distance: {fear_indicators[3] / fear_count} lid tightness: {fear_indicators[4] / fear_count} upper lid raiser: {fear_indicators[5] / fear_count}")
    else:
        print("No fear faces detected")
    if anger_count > 0:
        print(f"Anger - lid tightness: {anger_indicators[0] / anger_count} lip tightness: {anger_indicators[1] / anger_count} mid brow distance: {anger_indicators[2] / anger_count} upper lid raiser: {anger_indicators[3] / anger_count}")
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
        print(f"Filename : {image_path}")
        print(f"Actual Emotion: {actualEmotion}")
        emotion_determiner(faces, gray, actualEmotion)
    print("min: "+str(min(happy_cheek_raiser)) + "max: " + str(max(happy_cheek_raiser))+": happy cheek raiser")
    print("min: " + str(min(happy_lip_corner_puller)) + "max: " + str(max(happy_lip_corner_puller)) + ": happy lip corner puller")
    print("min: " + str(min(sad_mid_brow_distance)) + "max: " + str(max(sad_mid_brow_distance)) + ": sad mid brow distance")
    print("min: " + str(min(sad_lip_corner_depressor)) + "max: " + str(max(sad_lip_corner_depressor)) + ": sad lip corner depressor")
    print("min: " + str(min(sad_inner_brow_distance)) + "max: " + str(max(sad_inner_brow_distance)) + ": sad inner brow distance")
    print("min: " + str(min(anger_lip_tightness)) + "max: " + str(max(anger_lip_tightness)) + ": anger lip tightness")
    print("min: " + str(min(anger_lid_tightness)) + "max: " + str(max(anger_lid_tightness)) + ": anger lid tightness")
    print("min: " + str(min(anger_upper_lid_raiser)) + "max: " + str(max(anger_upper_lid_raiser)) + ": anger upper lid raiser")
    print("min: " + str(min(anger_mid_brow_distance)) + "max: " + str(max(anger_mid_brow_distance)) + ": anger mid brow distance")
    print("min: " + str(min(fear_jaw_drop)) + "max: " + str(max(fear_jaw_drop)) + ": fear jaw drop")
    print("min: " + str(min(fear_inner_brow_distance)) + "max: " + str(max(fear_inner_brow_distance)) + ": fear inner brow distance")
    print("min: " + str(min(fear_outer_brow_distance)) + "max: " + str(max(fear_outer_brow_distance)) + ": fear outer brow distance")
    print("min: " + str(min(fear_mid_brow_distance)) + "max: " + str(max(fear_mid_brow_distance)) + ": fear mid brow distance")
    print("min: " + str(min(fear_upper_lid_raiser)) + "max: " + str(max(fear_upper_lid_raiser)) + ": fear upper lid raiser")
    print("min: " + str(min(fear_lid_tightness)) + "max: " + str(max(fear_lid_tightness)) + ": fear lid tightness")
    print("min: " + str(min(disgust_nose_wrinkle)) + "max: " + str(max(disgust_nose_wrinkle)) + ": disgust nose wrinkle")
    print("min: " + str(min(disgust_lip_corner_depressor)) + "max: " + str(max(disgust_lip_corner_depressor)) + ": disgust lip corner depressor")
    print("min: " + str(min(disgust_lip_tightness)) + "max: " + str(max(disgust_lip_tightness)) + ": disgust lip tightness")