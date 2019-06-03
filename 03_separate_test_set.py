import os
import sys
from shutil import move
import pandas as pd


def main():
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    training_path = img_path + "/training/"
    if not os.path.exists(training_path):
        os.makedirs(training_path)

    test_path = img_path + "/test/"
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    dataframe = pd.read_csv("dataframe.csv")
    for index, data in dataframe.iterrows():
        img_id = str(data["objid"])
        img_label = str(data["label"])
        file = img_path + img_id + ".npy"
        test_data = []
        training_data = []
        if index % 10 == 0:
            test_file = test_path + img_id + ".npy"
            move(file, test_file)
            test_data.append([img_id, img_label, test_file])
        else:
            training_file = test_path + img_id + ".npy"
            move(file, training_file)
            training_data.append([img_id, img_label, training_file])

    col = ["objid", "label", "file"]
    training_df = pd.DataFrame(training_data, columns=col)
    test_df = pd.DataFrame(test_data, columns=col)
    training_df_path = img_path + "training_dataframe.csv"
    test_df_path = img_path + "test_dataframe.csv"
    training_df.to_csv(test_df_path, encoding='utf-8', index=False)
    test_df.to_csv(training_df_path, encoding='utf-8', index=False)


if __name__ == '__main__':
    main()
