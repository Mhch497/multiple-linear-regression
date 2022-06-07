import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt  # для графиков
import seaborn as sns  # для тепловой матрицы
from tkinter import *
from tkinter.filedialog import askopenfilename  # чтобы открыть файл через окошко


def cheddok(i, j):  # функция по определению связи по шкале Чеддока
    a = ''
    if x[i][j] < 0:
        x[i][j] = abs(x[i][j])
        a = 'обратная'
    elif x[i][j] > 0:
        a = 'прямая'
    if 0 <= x[i][j] < 0.1:
        print('Признаки ' + perem[i] + ' и ' + perem[j] + ' не коррелируемы')
    elif 0.1 <= x[i][j] < 0.3:
        print('Между ' + perem[i] + ' и ' + perem[j] + ' слабая, ' + a + ' связь')
    elif 0.3 <= x[i][j] < 0.5:
        print('Между ' + perem[i] + ' и ' + perem[j] + ' умеренная, ' + a + ' связь')
    elif 0.5 <= x[i][j] < 0.7:
        print('Между ' + perem[i] + ' и ' + perem[j] + ' средняя, ' + a + ' связь')
    elif 0.7 <= x[i][j] < 0.9:
        print('Между ' + perem[i] + ' и ' + perem[j] + ' сильная, ' + a + ' связь')
    elif 0.9 <= x[i][j] <= 1:
        print('Между ' + perem[i] + ' и ' + perem[j] + ' линейная, ' + a + '  связь')


def mtrx(i, m1):  # функция по перебору значений выше главной диагонали
    n, m = x.shape
    if i != n - 1:
        for j in range(m1, m):
            cheddok(i, j)
        i += 1
        mtrx(i, m1 + 1)


def R2zn(R2):  # функция по определению влияния факторов на результирующий признак
    if R2 == 0:
        print('\n Вариация результирующего признака Y полностью обусловлена действием факторов, не учтенных в модели')
    elif R2 == 1:
        print('\n Вариация результирующего признака Y полностью обусловлена действием факторов, учтенных в модели')
    elif 0 < R2 < 1:
        print('\n Изменение признака Y на ' + str(
            R2 * 100) + '% обуславливается факторными признаками, включеннными в модель')


global perem, x
perem = ('Y', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12')
root = Tk()
root.title("окно")
root.geometry("500x300")
lab = Label(root, text="Добавьте документ (в формате Excel)", font="Arial 18", fg="black", bg="white")
lab.grid()  # добавить кнопку ОК
filename = askopenfilename(parent=root)
root.mainloop()
dano = pd.read_excel(filename, header=0, usecols='A:L', index_col=0)
a = []  # названия столбцов
for i in range(dano.shape[1]):
    p = perem[i]
    a.append(p)
col = dano.columns = a  # переименовали столбцы
CoefCorrel = dano.corr()  # коэффициент корреляции
print(dano)
# коэффициенты по МНК
V = np.ones((dano.shape[0], 1))  # единичный вектор-столбец
y = dano[['Y']]
X = dano.drop(['Y'], axis=1)
X = np.hstack((V, X))
XT = X.transpose()
XTY = np.dot(XT, y)
XTX = np.dot(XT, X)
obr_XTX = np.linalg.inv(XTX)
b = np.dot(obr_XTX, XTY)

# тепловая матрица для коэффициента корреляции
corrmat = CoefCorrel
f, ax = plt.subplots(figsize=(9, 8))
s = sns.heatmap(corrmat, ax=ax, cmap="YlGnBu", linewidths=0.1)
print(s)

i, m1 = 0, 1
x = CoefCorrel
x = CoefCorrel.values
print('\n Матрица коэффициентов корреляции: \n')
print(CoefCorrel)
print('\n')
print('Анализ коэффициентов корреляции: \n')
mtrx(i, m1)
print('\n Матрица параметров: \n')
for i in range(len(b)):
    print(('b' + str(i) + '=' + str(b[i][0])), sep=',')

model = str(b[0][0])
for i in range(1, dano.shape[1]):
    if '-' in str(b[i][0]):
        model += str(b[i][0]) + ('*X' + str(i))
    else:
        model += '+' + str(b[i][0]) + ('*X' + str(i))
print('\nУравнение регрессии:\n')
print('y=' + model)
yT = y.transpose()
yTy = np.dot(yT, y)
bT = b.transpose()
bTXTY = np.dot(bT, XTY)
bTXTY = np.dot(bTXTY, -1)
eTe = yTy + bTXTY
y_sr = np.mean(y)
y_sr = np.array([[y_sr] * y.shape[0]])
y_sr = np.dot(y_sr, -1)
tss = (y + y_sr[0]) ** 2
TSS = np.sum(tss)
R2 = 1 - eTe[0][0] / TSS
r2 = float(R2[0])
R2zn(r2)
plt.show()
