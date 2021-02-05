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
        self.database.loadDatabase()


    def getFacematch(self, array) -> list:
        matches = []
        
        face_locations = face_recognition.face_locations(array)
        face_encodings = face_recognition.face_encodings(array, face_locations)
        for index, encoding in enumerate(face_encodings):
            results = face_recognition.compare_faces(self.database.Users, encoding)
            
            name = 'Unknown'
            print(results)
            if True in results:
                name = self.database.Users[results.index(True)].name
            
            print(name)
            matches.append([name, face_locations[index]])
        
        return matches


    def mainLoop(self) -> None:
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        
        while True:
            _, frame = cap.read()
            matches = self.getFacematch(frame)

            for name, location in matches:
                
                if self.config.draw_rectangle:
                    for (top, right, bottom, left) in [location]:
                        cv2.rectangle(frame, (left, top), (right, bottom), (250, 250, 250), 1)
            
            cv2.imshow(self.config.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()




if __name__ == "__main__":
    fr = FaceRecognition(os.path.abspath(os.path.dirname(__file__)))
    fr.mainLoop()