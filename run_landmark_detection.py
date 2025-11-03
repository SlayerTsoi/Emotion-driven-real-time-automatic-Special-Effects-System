import cv2
import numpy as np
import mediapipe as mp

# Load the sunglasses image with alpha channel
sunglasses_img = cv2.imread('images/sunglasses.png', cv2.IMREAD_UNCHANGED)

mp_face_mesh = mp.solutions.face_mesh

# Initialize MediaPipe Face Mesh
with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        # Create copies of the original image for drawing eye points and full face points
        points_image = image.copy()
        full_face_image = image.copy()

        try:
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get positions for the inner, outer, and center points of each eye
                    left_eye_inner = face_landmarks.landmark[133]  # Inner left eye
                    left_eye_outer = face_landmarks.landmark[33]   # Outer left eye
                    right_eye_inner = face_landmarks.landmark[362]  # Inner right eye
                    right_eye_outer = face_landmarks.landmark[263]  # Outer right eye

                    # Calculate positions in pixels
                    h, w, _ = image.shape
                    left_inner_pos = (int(left_eye_inner.x * w), int(left_eye_inner.y * h))
                    left_outer_pos = (int(left_eye_outer.x * w), int(left_eye_outer.y * h))
                    right_inner_pos = (int(right_eye_inner.x * w), int(right_eye_inner.y * h))
                    right_outer_pos = (int(right_eye_outer.x * w), int(right_eye_outer.y * h))

                    # Calculate the distance between inner and outer points to determine eye width
                    left_eye_width = int(np.linalg.norm(np.array(left_outer_pos) - np.array(left_inner_pos)))
                    right_eye_width = int(np.linalg.norm(np.array(right_outer_pos) - np.array(right_inner_pos)))
                    point_size = max(left_eye_width, right_eye_width) // 10  # Point size based on eye width

                    # Draw points on the eye points image
                    cv2.circle(points_image, left_inner_pos, point_size, (0, 0, 255), -1)  # Inner left eye
                    cv2.circle(points_image, left_outer_pos, point_size, (0, 0, 255), -1)  # Outer left eye
                    cv2.circle(points_image, (left_inner_pos[0] + (left_outer_pos[0] - left_inner_pos[0]) // 2, left_inner_pos[1]), point_size, (0, 255, 0), -1)  # Center left eye
                    cv2.circle(points_image, right_inner_pos, point_size, (0, 0, 255), -1)  # Inner right eye
                    cv2.circle(points_image, right_outer_pos, point_size, (0, 0, 255), -1)  # Outer right eye
                    cv2.circle(points_image, (right_inner_pos[0] + (right_outer_pos[0] - right_inner_pos[0]) // 2, right_inner_pos[1]), point_size, (0, 255, 0), -1)  # Center right eye

                    # Draw full face landmarks
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(full_face_image, (x, y), 2, (255, 0, 0), -1)  # Blue points for full face landmarks

                    # Calculate the width and height for the sunglasses
                    sunglasses_width = int(abs(left_outer_pos[0] - right_outer_pos[0]) * 1.1)  # Add some margin
                    sunglasses_height = int(sunglasses_width * sunglasses_img.shape[0] / sunglasses_img.shape[1])
                    sunglasses_resized = cv2.resize(sunglasses_img, (sunglasses_width, sunglasses_height))

                    # Calculate position for the sunglasses
                    sunglasses_x = int((left_outer_pos[0] + right_outer_pos[0]) / 2 - sunglasses_width / 2)
                    sunglasses_y = int((left_outer_pos[1] + right_outer_pos[1]) / 2 - sunglasses_height / 2)

                    # Overlay sunglasses onto the original image
                    for c in range(0, 3):
                        image[sunglasses_y:sunglasses_y + sunglasses_height, sunglasses_x:sunglasses_x + sunglasses_width, c] = \
                            sunglasses_resized[:, :, c] * (sunglasses_resized[:, :, 3] / 255.0) + \
                            image[sunglasses_y:sunglasses_y + sunglasses_height, sunglasses_x:sunglasses_x + sunglasses_width, c] * (1.0 - sunglasses_resized[:, :, 3] / 255.0)

            else:
                # Show "No people detected" message without flipping
                cv2.putText(image, "No people face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        except Exception as e:
            # Catch any error and display message
            cv2.putText(image, " ".format(str(e)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the images in separate windows
        cv2.imshow('Sunglasses Effect', image)       # Image with sunglasses
        cv2.imshow('Eye Points', points_image)       # Image with eye points
        cv2.imshow('Full Face Points', full_face_image)  # Image with full face points

        if cv2.waitKey(5) & 0xFF == 27:  # Exit on 'ESC' key
            break

    cap.release()
    cv2.destroyAllWindows()