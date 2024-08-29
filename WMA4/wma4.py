import cv2
from facerec import FaceRec

# Encode faces from a folder
sfr = FaceRec()
sfr.load_images_from_directory("images/")

# Camera setup
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_locations, face_names = sfr.detect_known_faces(frame)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 4)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Use 'q' to quit
            break
finally:
    cap.release()
    cv2.destroyAllWindows()