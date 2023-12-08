#Модуль builtin_data от Loginom
import builtin_data
from builtin_data import InputTable, OutputTable

import numpy as np
import pandas as pd
from builtin_pandas_utils import to_data_frame, prepare_compatible_table, fill_table

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


#Входной порт
if InputTable:
    # Создать pd.DataFrame по входному набору №1
    input_frame = to_data_frame(InputTable)

#Разделяем атрибуты и метки
X = input_frame.drop(input_frame.columns[-1], axis=1)
#Прогнозируемый показатель
Y = input_frame[input_frame.columns[-1]]

#Разделим случайно данные на обучающие и тестовые наборы. 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

#Обучаем модель на обучающих данных
classifier = DecisionTreeClassifier()
classifier.fit(X_train, Y_train)

#Прогноз на тестовых данных
Y_pred = classifier.predict(X_test)

#Слияние атрибутов и меток.
X_test = X_test.reset_index()
Y_pred = pd.Series(Y_pred)
df = pd.DataFrame()
df = pd.concat([df, Y_pred], axis=1)
X_test = pd.concat([X_test, df], axis=1)
print(X_test)

#Вывод результата в таблицу
prepare_compatible_table(OutputTable, X_test, with_index=False)
fill_table(OutputTable, X_test, with_index=False)  
