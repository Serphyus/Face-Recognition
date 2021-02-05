import os
import json
import cv2
import numpy as np
import face_recognition
from database import Database




class FaceRecognition:
    def __init__(self, path: str) -> object:
        with open(os.path.join(path, 'config.json'), 'r') as _file:
            self.config = type('config', (object,), json.load(_file)['FaceRecognitionConfig'])
        
        self.database = Database(path)
        self.database.encodeDatabase()
        self.database.loadDatabase()


    def getFacematch(self, array) -> list:
        matches = []
        
        face_locations = face_recognition.face_locations(array)
        face_encodings = face_recognition.face_encodings(array, face_locations)
        for index, encoding in enumerate(face_encodings):
            results = face_recognition.compare_faces(self.database.Users, encoding)
            
            name = 'Unknown'
            if True in results:
                name = self.database.Users[results.index(True)].name
            
            matches.append([name, face_locations[index]])
        
        return matches


    def mainLoop(self, camera_type='integrated') -> None:
        if camera_type == 'integrated':
            cap = cv2.VideoCapture(0)
        elif camera_type == 'external':
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        else:
            raise ValueError('camera_type argument must be integrated or external')
        
        if self.config.draw_name:
            fontFace = getattr(cv2, self.config.fontFace)
            fontScale = self.config.fontScale
            rectangleOffset = self.config.rectangleOffset
        
        while True:
            _, frame = cap.read()
            matches = self.getFacematch(frame)

            for name, location in matches:
                for (top, right, bottom, left) in [location]:
                    if self.config.draw_rectangle:
                        cv2.rectangle(frame, (left, top), (right, bottom), (250, 250, 250), 1)
            
                    if self.config.draw_name:
                        textSize, baseLine = cv2.getTextSize(name, fontFace, fontScale, 1)
                        if (right - left) < textSize[0]: right = (left + textSize[0] + rectangleOffset)

                        cv2.rectangle(frame, (left, bottom), (right, bottom + 25), (250, 250, 250), -1)
                        cv2.putText(frame, name, (left + rectangleOffset, bottom + 20), fontFace, fontScale, (0, 0, 0), 1, cv2.LINE_AA)
                    
            cv2.imshow(self.config.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()




if __name__ == "__main__":
    fr = FaceRecognition(os.path.abspath(os.path.dirname(__file__)))
    fr.mainLoop(camera_type='external')