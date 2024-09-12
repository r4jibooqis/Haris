from flask import Flask, jsonify, send_file
from flask_cors import CORS
import os

# Create Flask app
app = Flask(__name__)
CORS(app)

def read_face_risks_from_file():
    # Read face risks from a text file
    try:
        with open('face_risks.txt', 'r') as file:
            content = file.read()
            # Convert the text file content to a dictionary
            face_risks = eval(content)  # This assumes the file content is in a dictionary format
    except FileNotFoundError:
        face_risks = {}
    return face_risks

@app.route('/get_face_risks', methods=['GET'])
def get_face_risks():
    # Read face risks from the file
    face_risks = read_face_risks_from_file()
    
    # Convert face_risks dictionary to a list of dictionaries
    risks_list = [{"face_num": str(face_num), "risk": risk} for face_num, risk in face_risks.items()]

    # Return JSON response with image URL
    image_url = '/static/api_image.jpeg'  # URL where the image can be accessed
    
    return jsonify({
        'face_risks': risks_list,
        'image_url': image_url
    })

@app.route('/image')
def get_image():
    image_path = 'api_image.jpeg'
    
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/jpeg')
    else:
        return jsonify({'error': 'Image not found'}), 404

@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    # Ensure the 'static' folder exists and the image file is in it
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, port=5000)
