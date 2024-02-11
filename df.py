import cv2
from deepface import DeepFace
import os

db = './testImages/'

correct = 0
total = 0
happy = 0
sad = 0
anger = 0
fear = 0
neutral = 0
disgust = 0
right = False
print("Image,Actual_Emotion,Predicted_Emotion,Correct")

for file in os.listdir(db):
    image_path = os.path.join(db, file)  # Construct full file path
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue
    try:
        result = DeepFace.analyze(image_path, actions=['emotion'])
    except ValueError:
        result = DeepFace.analyze(image_path, actions=['emotion'],enforce_detection=False)

    total += 1

    # print result
    #print("Image name" +image_path)
    #print(result)

    actualEmotion = file[0]
    if actualEmotion == "h":
        actualEmotion = "happy"
        happy += 1
        # predictedEmotion = emotion_determiner(faces, gray, False)
    elif actualEmotion == "a":
        actualEmotion = "angry"
        anger += 1
        # predictedEmotion = emotion_determiner(faces, gray, False)
    elif actualEmotion == "n":
        actualEmotion = "neutral"
        neutral += 1
        # predictedEmotion = emotion_determiner(faces, gray, False)
    elif actualEmotion == "s":
        actualEmotion = "sad"
        sad += 1
        # predictedEmotion = emotion_determiner(faces, gray, False)
    elif actualEmotion == "f":
        actualEmotion = "fear"
        fear += 1
        # predictedEmotion = emotion_determiner(faces, gray, False)
    elif actualEmotion == "d":
        actualEmotion = "disgust"
        disgust += 1

    dominantEmotion = result[0]['dominant_emotion']
    #print("dominant emotion " + dominantEmotion)
    if dominantEmotion == actualEmotion:
        correct += 1

    print(f"{image_path},{actualEmotion},{dominantEmotion},{right}")

#print(f"Correct: {correct}")
#print(f"Total: {total}")