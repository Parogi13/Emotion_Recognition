import cv2
import dlib
import os
from jproperties import Properties

# Load in properties values
configs = Properties()
with open('config.properties', 'rb') as averages:
    configs.load(averages)

# Load the pre-trained facial landmark predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

# Set up FACES database
facesDB = "./FACES_database/"

# Function will be able to find magnitude of difference between two numbers and express as a percentage
def percent(number, avg):
    if number <= avg:
        return number/avg
    else:
        return ((2*avg - number)/avg)

# Function will determine the emotion
def emotion_determiner(shot, gray_scaled, show):

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

        # Happy
        happy_cheek_raiser_avg = float(configs.get("happy_cheek_raiser").data)
        happy_lip_corner_puller_avg = float(configs.get("happy_lip_corner_puller").data)
        '''happy_cheek_raiser = percent(cheek_raiser,happy_cheek_raiser_avg)
        happy_lip_corner_puller = percent(lip_corner_puller,happy_lip_corner_puller_avg)
        happyConfidence = (happy_cheek_raiser+happy_lip_corner_puller)/2
        #print(f"Happy confidence: {happyConfidence}")'''

        # Sad
        sad_mid_brow_distance_avg = float(configs.get("sad_mid_brow_distance").data)
        sad_inner_brow_distance_avg = float(configs.get("sad_inner_brow_distance").data)
        sad_lip_corner_depressor_avg = float(configs.get("sad_lip_corner_depressor").data)
        '''sad_mid_brow_distance = percent(mid_brow_distance, sad_mid_brow_distance_avg)
        sad_inner_brow_distance = percent(inner_brow_distance, sad_inner_brow_distance_avg)
        sad_lip_corner_depressor = percent(lip_corner_depressor,sad_lip_corner_depressor_avg)
        sadConfidence = (sad_lip_corner_depressor+sad_inner_brow_distance+sad_mid_brow_distance) / 3
        #print(f"Sad confidence: {sadConfidence}")'''

        # Disgust
        disgust_nose_wrinkle_avg = float(configs.get("disgust_nose_wrinkle").data)
        disgust_lip_tightness_avg = float(configs.get("disgust_lip_tightness").data)
        disgust_lip_corner_depressor_avg = float(configs.get("disgust_lip_corner_depressor").data)

        # Fear
        fear_jaw_drop_avg = float(configs.get("fear_jaw_drop").data)
        fear_inner_brow_distance_avg = float(configs.get("fear_inner_brow_distance").data)
        fear_mid_brow_distance_avg = float(configs.get("fear_mid_brow_distance").data)
        fear_outer_brow_distance_avg = float(configs.get("fear_outer_brow_distance").data)
        fear_lid_tightness_avg = float(configs.get("fear_lid_tightness").data)
        fear_upper_lid_raiser_avg = float(configs.get("fear_upper_lid_raiser").data)
        # Anger
        anger_lid_tightness_avg = float(configs.get("anger_lid_tightness").data)
        anger_lip_tightness_avg = float(configs.get("anger_lip_tightness").data)
        anger_mid_brow_distance_avg = float(configs.get("anger_mid_brow_distance").data)
        anger_upper_lid_raiser_avg = float(configs.get("anger_upper_lid_raiser").data)

        # Determine emotion based on indicators
        emotion = "Neutral"
        if (cheek_raiser < (happy_cheek_raiser_avg*1.2) and cheek_raiser > (happy_cheek_raiser_avg*.8)) and (lip_corner_puller > (happy_lip_corner_puller_avg*0.8) and lip_corner_puller < (happy_lip_corner_puller_avg*1.2)):
            emotion = "Happy"
        if (mid_brow_distance < (anger_mid_brow_distance_avg*1.2) and mid_brow_distance > (anger_mid_brow_distance_avg*0.8)) and (lid_tightness < (anger_lid_tightness_avg*1.2) and lid_tightness > anger_lip_tightness_avg*0.8) and (lip_tightness < anger_lip_tightness_avg*1.2 and lip_tightness > anger_lip_tightness_avg*0.8) and (upper_lid_raiser < anger_upper_lid_raiser_avg*1.2 and upper_lid_raiser > anger_upper_lid_raiser_avg*0.8):
            emotion = "Anger"
        if (inner_brow_distance < sad_inner_brow_distance_avg*1.2 and inner_brow_distance > sad_inner_brow_distance_avg*0.65) and (lip_corner_depressor < sad_lip_corner_depressor_avg*1.75 and lip_corner_depressor > sad_lip_corner_depressor_avg*0.4) and (mid_brow_distance < sad_mid_brow_distance_avg*1.4 and mid_brow_distance > sad_mid_brow_distance_avg*0.8):
            emotion = "Sad"
        if (nose_wrinkle > disgust_nose_wrinkle_avg*0.8 and nose_wrinkle < disgust_nose_wrinkle_avg*1.2) and (lip_corner_depressor < disgust_lip_corner_depressor_avg*2 and lip_corner_depressor > disgust_lip_corner_depressor_avg*0.8) and (lip_tightness > disgust_lip_tightness_avg*0.8 and lip_tightness < disgust_lip_tightness_avg*1.2):
            emotion = "Disgust"
        if (inner_brow_distance > fear_inner_brow_distance_avg*0.8 and inner_brow_distance < fear_inner_brow_distance_avg*1.2) and (outer_brow_distance < fear_outer_brow_distance_avg*1.2 and outer_brow_distance > fear_outer_brow_distance_avg*0.8) and (mid_brow_distance < fear_mid_brow_distance_avg*1.2 and mid_brow_distance > fear_mid_brow_distance_avg*0.8) and (lid_tightness > fear_lid_tightness_avg*0.8 and lid_tightness < fear_lid_tightness_avg*1.2) and (jaw_drop > fear_jaw_drop_avg*0.8 and jaw_drop < fear_jaw_drop_avg*1.2) and (upper_lid_raiser > fear_upper_lid_raiser_avg*0.8 and upper_lid_raiser < fear_upper_lid_raiser_avg*1.2):
            emotion = "Fear"

        print(f"Predicted Emotion: {emotion}")

        if show:
            # Returns and displays emotion
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 15)
            cv2.putText(image, "Emotion: " + emotion, (x - 100, (y+h) + 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0),8, cv2.LINE_AA)
            cv2.namedWindow("Emotion Recognition", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Emotion Recognition", 400, 600)
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
    emotion_determiner(faces, gray, True)
elif choice == "g":
    correct = 0
    total = 0
    happy = 0
    sad = 0
    anger = 0
    fear = 0
    neutral = 0
    disgust = 0
    wrongHappy = 0
    wrongSad = 0
    wrongAnger = 0
    wrongFear = 0
    wrongNeutral = 0
    wrongDisgust = 0
    predHappy = 0
    predSad = 0
    predAnger = 0
    predFear = 0
    predNeutral = 0
    predDisgust = 0

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
        predictedEmotion = ""
        predictedEmotion = emotion_determiner(faces, gray, False)

        actualEmotion = file[8]
        if actualEmotion == "h":
            actualEmotion = "Happy"
            happy += 1
            # predictedEmotion = emotion_determiner(faces, gray, False)
        elif actualEmotion == "a":
            actualEmotion = "Anger"
            anger += 1
            # predictedEmotion = emotion_determiner(faces, gray, False)
        elif actualEmotion == "n":
            actualEmotion = "Neutral"
            neutral += 1
            # predictedEmotion = emotion_determiner(faces, gray, False)
        elif actualEmotion == "s":
            actualEmotion = "Sad"
            sad += 1
            # predictedEmotion = emotion_determiner(faces, gray, False)
        elif actualEmotion == "f":
            actualEmotion = "Fear"
            fear += 1
            # predictedEmotion = emotion_determiner(faces, gray, False)
        elif actualEmotion == "d":
            actualEmotion = "Disgust"
            disgust += 1
            # predictedEmotion = emotion_determiner(faces, gray, False)
        print(f"Actual Emotion: {actualEmotion}")
        if predictedEmotion == "Happy":
            predHappy += 1
        elif predictedEmotion == "Anger":
            predAnger += 1
        elif predictedEmotion == "Neutral":
            predNeutral += 1
        elif predictedEmotion == "Sad":
            predSad += 1
        elif predictedEmotion == "Disgust":
            predDisgust += 1
        elif predictedEmotion == "Fear":
            predFear += 1
        if actualEmotion == predictedEmotion:
            correct += 1
        else:
            if actualEmotion == "Happy":
                wrongHappy += 1
            elif actualEmotion == "Anger":
                wrongAnger += 1
            elif actualEmotion == "Neutral":
                wrongNeutral += 1
            elif actualEmotion == "Sad":
                wrongSad += 1
            elif actualEmotion == "Disgust":
                wrongDisgust += 1
            elif actualEmotion == "Fear":
                wrongFear += 1
    print(f"Correct: {correct}")
    print(f"Total: {total}")
    print(f"Total happy: {happy}")
    print(f"Total sad: {sad}")
    print(f"Total anger: {anger}")
    print(f"Total fear: {fear}")
    print(f"Total neutral: {neutral}")
    print(f"Total disgust: {disgust}")
    print(f"Wrong happy: {wrongHappy}")
    print(f"Wrong sad: {wrongSad}")
    print(f"Wrong fear: {wrongFear}")
    print(f"Wrong disgust: {wrongDisgust}")
    print(f"Wrong neutral: {wrongNeutral}")
    print(f"Wrong anger: {wrongAnger}")
    print(f"Predicted happy: {predHappy}")
    print(f"Predicted sad: {predSad}")
    print(f"Predicted anger: {predAnger}")
    print(f"Predicted fear: {predFear}")
    print(f"Predicted disgust: {predDisgust}")
    print(f"Predicted neutral: {predNeutral}")
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()