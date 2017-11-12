import numpy as np
from PIL import Image
import os
from pathlib import Path
import pandas as pd
import xlrd as excel

# Ben MacMillan
# This class normalizes the retinal scan data for use in a convolutional neural network
class Data_Wrangler:

    def __init__(self):
        # initialize fields
        self.training_features = []
        self.training_labels = []
        self.testing_features = []
        self.testing_labels = []

        # for running locally (don't lol)
        local_path = "./Messidor/Base"
        local_path_grey = "./Messidor_Grey/Base"

        # for running on floydhub servers
        floyd_path_data = "/Scans"
        floyd_path_labels = "/Grades"

        # populate training fields
        for i in range(1, 11):

            # features
            for image in Path(floyd_path_data + "/Base" + str(i)).glob("*.tif"):
                img = Image.open(image)
                self.training_features.append(np.array(img))

            # labels
            label_sheet = excel.open_workbook(floyd_path_labels + "/Base" + str(i) + "/Base" + str(i) + ".xls")
            sheet = label_sheet.sheet_by_name('sheet')
            for num in sheet.col_values(2):
                if isinstance(num, float):
                    self.training_labels.append(num)

        # populate testing fields
        for i in range(11, 13):

            # features
            for image in Path(floyd_path_data + "/Base" + str(i)).glob("*.tif"):
                img = Image.open(image)
                self.testing_features.append(np.array(img))

            # labels
            label_sheet = excel.open_workbook(floyd_path_labels +  "/Base" + str(i) +  "/Base" + str(i) + ".xls")
            sheet = label_sheet.sheet_by_name('sheet')
            for num in sheet.col_values(2):
                if isinstance(num, float):
                    self.testing_labels.append(num)



    def populate_directory(self):
        # populate each base in the directory
        for i in range(1, 13):
            print("Populating directory: ./Messidor_Grey/Base" + str(i))
            files = []
            ix = 0

            # convert each image to greyscale for computational efficiency
            # resize each image to normalize dataset
            for image in Path("./Messidor/Base" + str(i)).glob("*.tif"):
                ix += 1
                img = Image.open(image).convert("L").resize((1440, 960))
                files.append(img)
                img.save(Path("./Messidor_Grey/Base" + str(i) + "/scan" + str(ix) + ".tif"))

    def get_training_features(self):
        return np.array(self.training_features, dtype=np.float32)

    def get_training_labels(self):
        return np.array(self.training_labels, dtype=np.float32)

    def get_testing_features(self):
        return np.array(self.testing_features, dtype=np.float32)

    def get_testing_labels(self):
        return np.array(self.testing_labels, dtype=np.float32)


if __name__ == "__main__":
    data_wrangler = Data_Wrangler()
    data_wrangler.populate_directory()
