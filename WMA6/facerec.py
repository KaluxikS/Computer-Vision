import face_recognition
import cv2
import os
import glob
import numpy as np

class FaceRec:
    def __init__(self, frame_resizing=0.25, mustache_path='images/wasy4.png'):
        self.known_faces = {}
        self.frame_resizing = frame_resizing
        self.mustache = cv2.imread(mustache_path, cv2.IMREAD_UNCHANGED)

    def add_mustache(self, frame, location):
        (top, right, bottom, left) = location
        face_width = right - left
        face_height = bottom - top

        mustache_width = face_width
        mustache_height = int(self.mustache.shape[0] * (mustache_width / self.mustache.shape[1]))
        resized_mustache = cv2.resize(self.mustache, (mustache_width, mustache_height))

        mustache_position = (left, bottom - int(face_height * 0.6))

        for i in range(resized_mustache.shape[0]):
            for j in range(resized_mustache.shape[1]):
                if resized_mustache[i, j][3] != 0:  
                    offset_y = mustache_position[1] + i
                    offset_x = mustache_position[0] + j
                    frame[offset_y, offset_x] = resized_mustache[i, j][:3]

        return frame


    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, locations)

        names = []
        for encoding, location in zip(encodings, locations):
            name = self._find_match(encoding)
            if name == "":
                scaled_location = (np.array(location) / self.frame_resizing).astype(int)
                frame = self.add_mustache(frame, scaled_location)
            names.append(name)

        locations = (np.array(locations) / self.frame_resizing).astype(int)
        return locations, names
    
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

    def _find_match(self, face_encoding):
        matches = face_recognition.compare_faces(list(self.known_faces.values()), face_encoding)
        distances = face_recognition.face_distance(list(self.known_faces.values()), face_encoding)
        best_match_index = np.argmin(distances)

        if matches[best_match_index]:
            return list(self.known_faces.keys())[best_match_index]
        return ""


