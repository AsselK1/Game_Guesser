import os.path
import json

import nltk
import pandas as pd
from os import listdir
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model import GamePredictor
from data.game_name import return_game_name


def generate_merged_data():
    path = os.path.join(os.path.dirname(__file__), 'data/data')

    files = listdir(path)

    file_dict = defaultdict(list)

    def meta_data_path(file: str) -> str:
        return os.path.join(os.path.dirname(__file__), f'data/data/Meta_Data/{file.split(".")[0]}_meta.json')

    def merged_data_path(file: str) -> str:
        return os.path.join(os.path.dirname(__file__), 'data/merged/{}'.format(file))

    for file in files:
        if not file:
            continue
        if os.path.isdir(os.path.join(path, file)):
            continue
        if file == ".DS_Store":
            continue
        file_target = open(meta_data_path(file))
        cur_json = json.load(file_target)
        file_dict[cur_json['stream_game_id']].append(file)

    for key in file_dict.keys():
        cur = file_dict[key]
        dfs = []
        for item in cur:
            try:
                dfs.append(pd.read_csv("{}/{}".format(path, item)))
            except Exception as E:
                print(E)
        try:
            combined = pd.concat(dfs, ignore_index=True)
            combined['stream_game_id'] = key
            combined.drop(['Time', 'User'], axis=1)
            combined.to_csv(merged_data_path(key), index=False)
        except Exception as E:
            print(E)


generate_merged_data()
var = {'Counter-Strike: Global Offensive', 'Magic: The Gathering', 'Grand Theft Auto V', 'World of Warcraft',
       'League of Legends', 'Call of Duty: Black Ops 4', 'Total War: Three Kingdoms', 'Apex Legends', 'Hearthstone',
       'Fortnite', 'Minecraft', 'Overwatch', 'Dead by Daylight', 'FIFA 19', "PLAYERUNKNOWN'S BATTLEGROUNDS", 'Dota 2',
       'StarCraft II', 'Diablo III: Reaper of Souls', 'Rocket League', "Tom Clancy's Rainbow Six: Siege"}


def train_model():
    path = os.path.join(os.path.dirname(__file__), 'data/merged')
    files = listdir(path)
    dataframes = []
    for file in files:
        try:
            df = pd.read_csv("{}/{}".format(path, file))
            dataframes.append(df)
        except Exception as E:
            print(E)
    combined_df = pd.concat(dataframes, ignore_index=True)
    x = combined_df.drop('stream_game_id', axis=1)
    y = combined_df['stream_game_id']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    ids = os.listdir(path) 
    train_list = []
    test_list = []
    train_df = x_train
    train_df.insert(3, "stream_game_id", y_train)
    test_df = x_test
    test_df.insert(3,  "stream_game_id", y_test)
    for id in ids:
        train_list.append(train_df[train_df["stream_game_id"] == int(id)])
        test_list.append(test_df[test_df["stream_game_id"] == int(id)])
    game_predictor = GamePredictor()
    game_predictor.train_classifier(train_list)
    game_predictor.create_game_keywords(train_df['Message'])
    print(return_game_name(str(test_list[0]['stream_game_id'].iloc[0])))
    print(return_game_name(game_predictor.predict_game(test_list[0]["Message"])))
    print(game_predictor.evaluate(test_set=test_list))

train_model()
