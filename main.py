import pandas as pd
import numpy as np

test = pd.read_csv('./csgo_dataset/test.csv', escapechar='`', low_memory=False)
players_feats = pd.read_csv('./csgo_dataset/players_feats.csv', escapechar='`', low_memory=False)
train = pd.read_csv('./csgo_dataset/train.csv', escapechar='`', low_memory=False)

players_feats['kd_ratio'] = players_feats[['p1_kd_ratio', 'p2_kd_ratio', 'p3_kd_ratio', 'p4_kd_ratio', 'p5_kd_ratio']].sum(axis=1) # вычисляем сумму рейтингов игроков команды

train = train.merge(players_feats, how='left', left_on=['team1_id', 'map_id', 'map_name'], right_on=['team_id', 'map_id', 'map_name']) # присоединяем статы игроков team1
train = train.merge(players_feats, how='left', left_on=['team2_id', 'map_id', 'map_name'], right_on=['team_id', 'map_id', 'map_name']) # присоединяем статы игроков team2

train = train.drop(train[(train['kd_ratio_x'] == 0) | (train['kd_ratio_y'] == 0)].index)  # удаляем статы матчей команд с нулевым рейтингом, считая, что они не должны влиять на статистику

train['kd_ratio_diff'] = train['kd_ratio_y'] - train['kd_ratio_x'] # разница в классе между командами

max_diff = max(train['kd_ratio_diff'].max(), abs(train['kd_ratio_diff'].min())) # максимальная известная разница в классе между командами

# print(max_diff)

test = test.merge(players_feats, how='left', left_on=['team1_id', 'map_id', 'map_name'], right_on=['team_id', 'map_id', 'map_name']) # присоединяем статы игроков team1
test = test.merge(players_feats, how='left', left_on=['team2_id', 'map_id', 'map_name'], right_on=['team_id', 'map_id', 'map_name']) # присоединяем статы игроков team2

test['kd_ratio_diff'] = test['kd_ratio_y'] - test['kd_ratio_x'] # разница в классе между командами

test['chance_to_win'] = 0.5 + test['kd_ratio_diff'] / (2 * max_diff)

for i in test.index:
    if test['chance_to_win'][i] > 1: # проверка вероятности на случай, если разница в классе выше максимально известной ранее
        test['chance_to_win'][i] = 1
    elif test['chance_to_win'][i] < 0:
        test['chance_to_win'][i] = 0
    if (test['kd_ratio_x'][i] == 0) or (test['kd_ratio_y'][i]) == 0: # если рейтинг команды не подсчитан, вероятность победы вычислить невозможно
        test['chance_to_win'][i] = np.nan
    test['chance_to_win'] = round(test['chance_to_win'], 4)

test = test[['index', 'team1_id', 'team2_id', 'map_id', 'map_name', 'chance_to_win']]
test.to_csv('./csgo_dataset/test_result.csv', encoding='utf-8')
