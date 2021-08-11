import os.path

class FileStorage:

    def __init__(self, file_name):
        self.path = "storage/" + file_name
        if os.path.isfile(self.path):
            self.file = open(self.path, "r")
        else:
            self.file = open(self.path, "w+")

    def get(self):
        return self.file.read()

    def refresh(self):
        """

        :rtype: FileStorage
        """
        self.file = open(self.path, "r")
        return self
