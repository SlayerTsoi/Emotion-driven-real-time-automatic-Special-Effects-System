import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import imutils

# Path configuration
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.hdf5'

# Load special effect pictures
sunglasses_img = cv2.imread('images/sunglasses.png', cv2.IMREAD_UNCHANGED)
effect_img = cv2.imread('images/angry.png', cv2.IMREAD_UNCHANGED) 
scared_img = cv2.imread('images/scared.png', cv2.IMREAD_UNCHANGED)
tear_img = cv2.imread('images/sad.png', cv2.IMREAD_UNCHANGED)
surprise_img = cv2.imread('images/surprise.png', cv2.IMREAD_UNCHANGED)

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh

# Load the model
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Adjust the size
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, 
                                              minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        
        # Convert color space
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        # Initialize all display windows
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frame_clone = frame.copy()           # Main window: with frame and no special effects
        emotion_detect_frame = frame.copy()  # Special effects window: frameless with special effects
        landmark_frame = frame.copy()        # Key point display window
        
        if len(faces) > 0:
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            
            # draw probability bars
            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            
            # Special effects application and key point drawing
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    
                    # Draw all landmark points (gray dots)
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(landmark_frame, (x, y), 1, (100, 100, 100), -1)
                    
                    # Draw key points based on different emotions
                    if label == "happy":
                        # Draw sunglasses related points
                        points = [133, 33, 362, 263]  # Inner and outer corners of eyes
                        for idx in points:
                            lm = face_landmarks.landmark[idx]
                            x = int(lm.x * w)
                            y = int(lm.y * h)
                            cv2.circle(landmark_frame, (x, y), 5, (0, 255, 0), -1)
                            cv2.putText(landmark_frame, str(idx), (x+5, y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                        
                        # Apply the Sunglasses effect
                        left_eye_inner = face_landmarks.landmark[133]
                        left_eye_outer = face_landmarks.landmark[33]
                        right_eye_inner = face_landmarks.landmark[362]
                        right_eye_outer = face_landmarks.landmark[263]
                        
                        sunglasses_width = int(abs(left_eye_outer.x * w - right_eye_outer.x * w) * 1.1)
                        sunglasses_height = int(sunglasses_width * sunglasses_img.shape[0] / sunglasses_img.shape[1])
                        sunglasses_resized = cv2.resize(sunglasses_img, (sunglasses_width, sunglasses_height))
                        
                        sunglasses_x = int((left_eye_outer.x * w + right_eye_outer.x * w) / 2 - sunglasses_width / 2)
                        sunglasses_y = int((left_eye_outer.y * h + right_eye_outer.y * h) / 2 - sunglasses_height / 2)
                        
                        # Transparent Overlay
                        alpha_s = sunglasses_resized[:, :, 3] / 255.0
                        alpha_l = 1.0 - alpha_s
                        for c in range(0, 3):
                            emotion_detect_frame[sunglasses_y:sunglasses_y+sunglasses_height, 
                                              sunglasses_x:sunglasses_x+sunglasses_width, c] = \
                                (alpha_s * sunglasses_resized[:, :, c] + 
                                 alpha_l * emotion_detect_frame[sunglasses_y:sunglasses_y+sunglasses_height, 
                                                              sunglasses_x:sunglasses_x+sunglasses_width, c])

                    elif label == "angry":
                        # Draw anger related points
                        points = [10, 152]  # Forehead and Chin
                        for idx in points:
                            lm = face_landmarks.landmark[idx]
                            x = int(lm.x * w)
                            y = int(lm.y * h)
                            cv2.circle(landmark_frame, (x, y), 5, (0, 0, 255), -1)
                            cv2.putText(landmark_frame, str(idx), (x+5, y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                        
                        # Apply angry effects
                        forehead = face_landmarks.landmark[10]
                        chin = face_landmarks.landmark[152]
                        
                        head_height = int((chin.y - forehead.y) * h * 1.6)
                        head_width = int(head_height * effect_img.shape[1] / effect_img.shape[0])
                        effect_resized = cv2.resize(effect_img, (head_width, head_height))
                        
                        effect_x = int(forehead.x * w - head_width/2)
                        effect_y = int(forehead.y * h - head_height/2)
                        
                        # Bounds Checking
                        effect_y = max(effect_y, 0)
                        effect_x = max(effect_x, 0)
                        
                        # Transparent Overlay
                        if effect_y + head_height < h and effect_x + head_width < w:
                            alpha = effect_resized[:, :, 3] / 255.0
                            for c in range(3):
                                emotion_detect_frame[effect_y:effect_y+head_height, effect_x:effect_x+head_width, c] = \
                                    effect_resized[:, :, c] * alpha + \
                                    emotion_detect_frame[effect_y:effect_y+head_height, effect_x:effect_x+head_width, c] * (1 - alpha)

                    elif label == "scared":
                        # Draw fear related points
                        points = [151, 162, 389]  # Center of forehead, left and right temples
                        for idx in points:
                            lm = face_landmarks.landmark[idx]
                            x = int(lm.x * w)
                            y = int(lm.y * h)
                            cv2.circle(landmark_frame, (x, y), 5, (255, 0, 255), -1)
                            cv2.putText(landmark_frame, str(idx), (x+5, y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                        
                        # Apply the fear effect
                        forehead_center = face_landmarks.landmark[151]
                        left_temple = face_landmarks.landmark[162]
                        right_temple = face_landmarks.landmark[389]
                        
                        effect_width = int(abs(right_temple.x * w - left_temple.x * w) * 1.3)
                        effect_height = int(effect_width * scared_img.shape[0] / scared_img.shape[1])
                        effect_resized = cv2.resize(scared_img, (effect_width, effect_height))
                        
                        effect_x = int(forehead_center.x * w - effect_width // 2)
                        effect_y = int(forehead_center.y * h - effect_height // 2 + h * 0.15)
                        
                        # Bounds Checking
                        effect_x = max(effect_x, 0)
                        effect_y = max(effect_y, 0)
                        if effect_y + effect_height > h:
                            effect_height = h - effect_y
                            effect_resized = cv2.resize(scared_img, (effect_width, effect_height))
                        
                        # Transparent Overlay
                        alpha_s = effect_resized[:, :, 3] / 255.0
                        alpha_l = 1.0 - alpha_s
                        for c in range(0, 3):
                            try:
                                emotion_detect_frame[effect_y:effect_y+effect_height, effect_x:effect_x+effect_width, c] = \
                                    (alpha_s * effect_resized[:, :, c] + 
                                     alpha_l * emotion_detect_frame[effect_y:effect_y+effect_height, effect_x:effect_x+effect_width, c])
                            except:
                                pass

                    elif label == "sad":
                        # Plot sadness related points
                        points = [145, 159, 374, 386, 133, 33, 362, 263]  # 眼睛上下缘
                        for idx in points:
                            lm = face_landmarks.landmark[idx]
                            x = int(lm.x * w)
                            y = int(lm.y * h)
                            cv2.circle(landmark_frame, (x, y), 5, (255, 255, 0), -1)
                            cv2.putText(landmark_frame, str(idx), (x+5, y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                        
                        # Apply sad effects
                        left_eye_under = face_landmarks.landmark[145]
                        left_eye_lid = face_landmarks.landmark[159]
                        right_eye_under = face_landmarks.landmark[374]
                        right_eye_lid = face_landmarks.landmark[386]
                        
                        # Left Tears
                        l_width = int(abs(face_landmarks.landmark[33].x * w - face_landmarks.landmark[133].x * w) * 1.8)
                        l_height = int(l_width * tear_img.shape[0] / tear_img.shape[1])
                        l_x = int(left_eye_under.x * w - l_width/2)
                        l_y = int((left_eye_under.y * 0.3 + left_eye_lid.y * 0.7) * h)
                        
                        # Right Tears
                        r_width = int(abs(face_landmarks.landmark[263].x * w - face_landmarks.landmark[362].x * w) * 1.8)
                        r_height = int(r_width * tear_img.shape[0] / tear_img.shape[1])
                        r_x = int(right_eye_under.x * w - r_width/2)
                        r_y = int((right_eye_under.y * 0.3 + right_eye_lid.y * 0.7) * h)

                        # Applying special effects
                        for x, y, width, height in [(l_x, l_y, l_width, l_height), (r_x, r_y, r_width, r_height)]:
                            tear_resized = cv2.resize(tear_img, (width, height))
                            
                            # Bounds Checking
                            y = max(y, 0)
                            x = max(x, 0)
                            if y + height > h:
                                height = h - y
                                tear_resized = cv2.resize(tear_img, (width, height))
                            if x + width > w:
                                width = w - x
                                tear_resized = cv2.resize(tear_img, (width, height))
                            
                            # Transparent Overlay
                            alpha = tear_resized[:, :, 3] / 255.0
                            for c in range(3):
                                try:
                                    emotion_detect_frame[y:y+height, x:x+width, c] = \
                                        tear_resized[:, :, c] * alpha + \
                                        emotion_detect_frame[y:y+height, x:x+width, c] * (1 - alpha)
                                except:
                                    pass

                    elif label == "surprised":
                        # Draw surprise related points
                        points = [10, 152]  # Forehead and Chin
                        for idx in points:
                            lm = face_landmarks.landmark[idx]
                            x = int(lm.x * w)
                            y = int(lm.y * h)
                            cv2.circle(landmark_frame, (x, y), 5, (0, 255, 255), -1)
                            cv2.putText(landmark_frame, str(idx), (x+5, y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                        
                        # Apply surprise effects
                        forehead = face_landmarks.landmark[10]
                        chin = face_landmarks.landmark[152]
                        
                        head_height = int((chin.y - forehead.y) * h * 1.8)
                        head_width = int(head_height * surprise_img.shape[1] / surprise_img.shape[0])
                        effect_resized = cv2.resize(surprise_img, (head_width, head_height))
                        
                        effect_x = int(forehead.x * w - head_width/2)
                        effect_y = int(forehead.y * h - head_height * 1.2)
                        
                        # Bounds Checking
                        effect_y = max(effect_y, 0)
                        effect_x = max(effect_x, 0)
                        
                        # Transparent Overlay
                        if effect_y + head_height < h and effect_x + head_width < w:
                            alpha = effect_resized[:, :, 3] / 255.0
                            for c in range(3):
                                emotion_detect_frame[effect_y:effect_y+head_height, effect_x:effect_x+head_width, c] = \
                                    effect_resized[:, :, c] * alpha + \
                                    emotion_detect_frame[effect_y:effect_y+head_height, effect_x:effect_x+head_width, c] * (1 - alpha)

            # Draw the main window elements
            cv2.putText(frame_clone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.rectangle(frame_clone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

            # Draw special effects window label
            cv2.putText(emotion_detect_frame, label, (fX, fY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
        else:
            continue
        
        
        cv2.imshow('Main Window (Bounding Box)', frame_clone)
        cv2.imshow('Effect Window (No Box)', emotion_detect_frame)
        cv2.imshow("Probabilities", canvas)
        cv2.imshow("Landmark Points", landmark_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.release()
    cv2.destroyAllWindows()