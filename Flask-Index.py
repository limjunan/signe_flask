
# Flask dependencies
from flask import Flask, render_template, Response
app = Flask(__name__)
# OpenCV
import cv2
# mediapipe dependencies and init
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands



camera = cv2.VideoCapture(0)

def gen_frames():  
    with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
        while True:
            success, frame = camera.read()  # read the camera frame
            if not success:
                print('ignoring empty frame')
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False
            results = hands.process(frame)

            # Draw the hand annotations on the image.
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


#  index home page
@app.route("/")
@app.route("/index")
def home():
    return render_template('index.html')



#  about page
@app.route("/video_feed")
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ ==  '__main__':
    app.run(debug=True)