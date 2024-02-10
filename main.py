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
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        # Indicators from the Facial Action Coding System
        outer_brow_distance = abs(landmarks.part(17).y - landmarks.part(36).y)/(h)
        inner_brow_distance = abs(landmarks.part(20).y - landmarks.part(38).y)/(h)
        mid_brow_distance = abs(landmarks.part(19).y - landmarks.part(37).y)/(h)
        lid_tightness = abs(landmarks.part(41).y - landmarks.part(37).y)/(h)
        lip_corner_depressor = abs(landmarks.part(48).x - landmarks.part(54).x)/(w)
        lip_corner_puller = abs(landmarks.part(54).y - landmarks.part(48).y)/(h)
        lip_tightness = abs(landmarks.part(51).y - landmarks.part(57).y)/(h)
        jaw_drop = abs(landmarks.part(57).y - landmarks.part(8).y)/(h)
        nose_wrinkle = abs(landmarks.part(31).x - landmarks.part(35).x)/(w)
        cheek_raiser = abs(landmarks.part(1).y - landmarks.part(4).y)/(h)
        upper_lid_raiser = abs(landmarks.part(19).y - landmarks.part(41).y)/(h)

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
        if (cheek_raiser < (0.37463335754342814*1.3) and cheek_raiser > (0.37463335754342814*0.8)) and (lip_corner_puller > (0.015152883642177718*0) and lip_corner_puller < (0.015152883642177718*2.25)):
            emotion = "Happy"
        if (mid_brow_distance < (0.08960082063968598*1.3) and mid_brow_distance > (0.08960082063968598*0.6)) and (lid_tightness < (0.04096565598249651*1.6) and lid_tightness > 0.04096565598249651*0.7) and (lip_tightness < 0.05944637349849057*2.1 and lip_tightness > 0.05944637349849057*0.03) and (upper_lid_raiser < 0.1305664766221825*1.3 and upper_lid_raiser > 0.1305664766221825*0.7):
            emotion = "Anger"
        if (inner_brow_distance < 0.10798011548150865*1.2 and inner_brow_distance > 0.10798011548150865*0.8) and (lip_corner_depressor < 0.3651929079070056*1.2 and lip_corner_depressor > 0.3651929079070056*0.8) and (mid_brow_distance < 0.12011136183002478*1.2 and mid_brow_distance > 0.12011136183002478*0.8):
            emotion = "Sad"
        if (nose_wrinkle > 0.21200304711685555*0.8 and nose_wrinkle < 0.21200304711685555*1.2) and (lip_corner_depressor < 0.34682658188341575*1.2 and lip_corner_depressor > 0.34682658188341575*0.8) and (lip_tightness > 0.11492076682292701*0.8 and lip_tightness < 0.11492076682292701*1.2):
            emotion = "Disgust"
        if (inner_brow_distance > 0.1413795034423165*0.8 and inner_brow_distance < 0.1413795034423165*1.2) and (outer_brow_distance < 0.07223343417369775*1.2 and outer_brow_distance > 0.07223343417369775*0.8) and (mid_brow_distance < 0.14802722709155844*1.2 and mid_brow_distance > 0.14802722709155844*0.8) and (lid_tightness > 0.06546390403684983*0.8 and lid_tightness < 0.06546390403684983*1.2) and (jaw_drop > 0.25328672872909336*0.8 and jaw_drop < 0.25328672872909336*1.2) and (upper_lid_raiser > 0.25328672872909336*0.8 and upper_lid_raiser < 0.25328672872909336*1.2):
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
            #predictedEmotion = emotion_determiner(faces, gray, False)
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