# Install necessary libraries
#!pip install mediapipe opencv-python

# Import libraries
import cv2
import mediapipe as mp

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.2, min_tracking_confidence=0.2)

# Function to resize frame to fit within screen dimensions
def resize_frame(frame):
    # Get screen dimensions
    screen_width = 1366  # Adjust this value according to your screen resolution
    screen_height = 768  # Adjust this value according to your screen resolution
    
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Calculate scaling factors
    scale_width = screen_width / frame_width
    scale_height = screen_height / frame_height
    
    # Choose the minimum scaling factor to ensure the entire frame fits within the screen
    scale_factor = min(scale_width, scale_height)
    
    # Resize the frame
    resized_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
    
    return resized_frame

# Function to detect rope jumping gesture in video
def detect_rope_jumping(video_path):
    cap = cv2.VideoCapture(video_path)
    rope_jumping_detected = False  # Flag to indicate if rope jumping is detected
    
    consecutive_frames_with_gesture = 0
    required_consecutive_frames = 10  # Adjust this value as needed
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to fit within screen dimensions
        frame = resize_frame(frame)
        
        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process image with MediaPipe Holistic
        results = holistic.process(rgb_frame)
        
        # Check if pose landmarks are detected
        if results.pose_landmarks:
            # Get the y-coordinate of the wrists landmarks
            left_wrist_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y
            right_wrist_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y
            
            # If wrists landmarks y-coordinates are above a certain threshold, increment consecutive_frames_with_gesture
            if left_wrist_y > 0.2 and right_wrist_y > 0.2:
                consecutive_frames_with_gesture += 1
            else:
                consecutive_frames_with_gesture = 0
                
            # If consecutive_frames_with_gesture reaches required_consecutive_frames, set rope_jumping_detected flag
            if consecutive_frames_with_gesture >= required_consecutive_frames:
                rope_jumping_detected = True
        
        # Draw detection result onto the frame
        if rope_jumping_detected:
            cv2.putText(frame, "DETECTED", (frame.shape[1] - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)  # Changed text position to top right corner
        else:
            cv2.putText(frame, "NOT DETECTED", (frame.shape[1] - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  # Changed text position to top right corner
        
        # Display the frame
        cv2.imshow('Rope Jumping Gesture Detection', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Run the function with your video file
video_path = "E:/Projects/Gesture Detection in Video Sequences/video.mp4"
detect_rope_jumping(video_path)
