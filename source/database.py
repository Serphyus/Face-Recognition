import os
import json
import pickle
import numpy as np
import face_recognition
from random import random
from contextlib import suppress




class Database:
    class User(list):
        def __init__(self, user_data: dict, encoding: list) -> object:
            if type(user_data['name']) != str:
                raise KeyError('user_data[\'name\'] must be str')

            elif not isinstance(encoding, np.ndarray):
                raise TypeError('encoding argument must be a ndarray')

            super().__init__(encoding)

            self.available_attrs = user_data.keys()
            for attr, content in user_data.items():
                if hasattr(self, attr):
                    raise AttributeError(f'{attr} already exists')
                setattr(self, attr, content)

        
        def get_data(self) -> str:
            ret = {}
            for key in self.available_attrs:
                ret[key] = getattr(self, key)
            return ret


    def __init__(self, path: str) -> object:
        self.path = path
        self.Users = []
        
        with open(os.path.join(self.path, 'config.json'), 'r') as _file:
            self.config = self._createObject('config', json.load(_file)['DatabaseConfig'])

        self.folders = self._createObject('folders',
            {
                'root': os.path.join(self.path, 'database'),
                'raw': os.path.join(self.path, 'database/raw'),
                'encoded': os.path.join(self.path, 'database/encoded'),
                'encoded_users': os.path.join(self.path, 'database/encoded/users')
            }
        )

        self._setup()
        self._getMetadata()


    def _setup(self) -> None:
        database_folders = [getattr(self.folders, folder) for folder in dir(self.folders) if not folder.startswith('__')]
        database_folders.sort()
        
        with suppress(FileExistsError):
            for folder in database_folders:
                current_folder = os.path.join(self.path, folder)
            
                if os.path.isdir(current_folder):
                    raise FileExistsError(f'{current_folder} already exists')

                os.mkdir(current_folder)
                print(f'creating: {current_folder}')

            print('\nDatabase Successfully Setup')
    

    def _createObject(self, name: str, d: dict) -> object:
        obj = type(name, (object,), d)
        
        return obj


    def _addUser(self, **user_data: dict) -> None:
        self.Users.append(
            self.User(**user_data)
        )


    def _getMetadata(self) -> None:
        self.metadata = self._createObject('metadata',
            {
                'pre_encoded': {}
            }
        )
        
        with suppress(FileNotFoundError, EOFError):
            with open(os.path.join(self.folders.encoded, 'meta.pkl'), 'rb') as _file:
                self.metadata = self._createObject('metadata', pickle.load(_file))


    def _updateMetadata(self) -> None:
        folder = os.path.join(self.path, self.folders.encoded)
        with open(os.path.join(folder, 'meta.pkl'), 'wb') as _file:
            _metadata = {}
            for attr in dir(self.metadata):
                if not attr.startswith('__'):
                    _metadata[attr] = getattr(self.metadata, attr)

            _file.truncate()
            pickle.dump(_metadata, _file)


    def _checkPreEncoded(self) -> None:
        current_encodings = os.listdir(self.folders.encoded_users)
        for encoded_file in current_encodings:
            if encoded_file[:encoded_file.index('.dat')] not in self.metadata.pre_encoded:
                print('[!] Unrecognizable file: %s' % encoded_file)
                os.remove(os.path.join(self.folders.encoded_users, encoded_file))

        for uid in list(self.metadata.pre_encoded):
            if not str(uid + '.dat') in current_encodings:
                print('[!] Unable to locate: %s.dat' % uid)
                self.metadata.pre_encoded.pop(uid)


    def _encodeUser(self, user_folder: str) -> None:
        path = os.path.join(self.folders.raw, user_folder)
        
        with open(os.path.join(path, 'user.json'), 'r') as _file:
            user_data = json.load(_file)

        face_image = face_recognition.load_image_file(os.path.join(path, 'face.jpg'))
        face_encoding = face_recognition.face_encodings(face_image, **self.config.faceEncodingArgs)[0]

        user = {
            'user_data': user_data,
            'encoding': face_encoding
        }

        uid = '{%s-%s-%s}' % (str(random())[-5:], str(random())[-5:], str(random())[-5:])
        with open(os.path.join(self.folders.encoded_users, '%s.dat' % uid), 'wb') as _file:
            pickle.dump(user, _file)

        self.metadata.pre_encoded[uid] = user_folder


    def encodeDatabase(self, output=True) -> None:
        self._checkPreEncoded()
        for user_folder in os.listdir(self.folders.raw):
            if user_folder not in self.metadata.pre_encoded.values():
                self._encodeUser(user_folder)
        self._updateMetadata()


    def loadDatabase(self) -> None:
        for filename in os.listdir(self.folders.encoded_users):
            with open(os.path.join(self.folders.encoded_users, filename), 'rb') as _file:
                self._addUser(**pickle.load(_file))




if __name__ == "__main__":
    d = Database(os.path.abspath(os.path.dirname(__file__)))
    d.encodeDatabase()