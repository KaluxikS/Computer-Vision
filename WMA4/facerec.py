import face_recognition
import cv2
import os
import glob
import numpy as np

class FaceRec:
    def __init__(self, frame_resizing=0.25):
        self.known_faces = {}
        self.frame_resizing = frame_resizing

    def load_images_from_directory(self, directory_path):
        image_files = glob.glob(os.path.join(directory_path, "*.*"))
        print(f"{len(image_files)} encoding images found.")

        for image_path in image_files:
            name = os.path.splitext(os.path.basename(image_path))[0]
            encoding = self._encode_face(image_path)
            if encoding is not None:
                self.known_faces[name] = encoding
        print("Encoding images loaded")

    def _encode_face(self, image_path):
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_image)
        return encodings[0] if encodings else None

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, locations)

        names = []
        for encoding in encodings:
            name = self._find_match(encoding)
            names.append(name)

        locations = (np.array(locations) / self.frame_resizing).astype(int)
        return locations, names

    def _find_match(self, face_encoding):
        matches = face_recognition.compare_faces(list(self.known_faces.values()), face_encoding)
        distances = face_recognition.face_distance(list(self.known_faces.values()), face_encoding)
        best_match_index = np.argmin(distances)

        if matches[best_match_index]:
            return list(self.known_faces.keys())[best_match_index]
        return "Unknown"
