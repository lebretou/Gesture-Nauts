from flask import Flask, render_template, Response, jsonify
import cv2
from process import process_frame

app = Flask(__name__)

# Video source and capture setup
video_source = 0  # Use the default camera
vid = cv2.VideoCapture(video_source)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height

# State to control processing
processing_enabled = False
current_gesture = {'gesture': "Gesture not detected...", 'emoji': "❓"}

# Emoji map
emoji_map = {
    "Call": "📞",
    "Dislike": "👎",
    "Fist": "✊",
    "Four": "4️⃣",
    "Like": "👍",
    "Mute": "🔇",
    "Ok": "👌",
    "One": "👆",
    "Palm": "🖐️",
    "Peace": "✌️",
    "Peace_inverted": "✌️",
    "Rock": "🤘",
    "Stop": "🛑",
    "Stop_inverted": "🛑",
    "Three": "3️⃣",
    "Three 2": "3️⃣",
    "Two_up": "🤞",
    "Two_up_inverted": "🤞",
    "Move": "👉",
}

def gen_frames():
    global processing_enabled, current_gesture
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # Mirror the image
        frame = cv2.resize(frame, (640, 480))  # Resize frame to fit the canvas
        if processing_enabled:
            frame, label = process_frame(frame)  # Process the frame if enabled
            emoji = emoji_map.get(label, "❓")
            if label not in emoji_map:
                current_gesture = {'gesture': "Gesture not detected...", 'emoji': "❓"}
            else:
                current_gesture = {'gesture': label, 'emoji': emoji}
        else:
            current_gesture = {'gesture': "Gesture not detected...", 'emoji': "❓"}

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/current_gesture')
def get_current_gesture():
    print(current_gesture)
    return jsonify(current_gesture)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_processing')
def toggle_processing():
    global processing_enabled
    processing_enabled = not processing_enabled
    return jsonify({
        'processing': processing_enabled,
        'message': "Disable Processing" if processing_enabled else "Enable Processing"
    })


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)