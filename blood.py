import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset_treino = pd.read_csv('dados/data_treino_eliminacao_alcool_mulher.csv', index_col=0)

vars_ind = ['BrAC (mg/L/h)']
var_dep = 'BAC (g/L/h)'

# print("## DATASET ##\n{}".format(dataset_treino))

x_treino, x_val, y_treino, y_val = train_test_split(dataset_treino[vars_ind], dataset_treino[var_dep], test_size=0.10)

# print("Conjunto de treino X:\n{}".format(x_treino))
# print("\nConjunto de validação X:\n{}".format(x_val))
# print("\nConjunto de treino Y:\n{}".format(y_treino))
# print("\nConjunto de validação Y:\n{}".format(y_val))

lr = LinearRegression()
lr.fit(x_treino, y_treino)

x = x_val.iloc[0]
y = y_val.iloc[0]
y_hat = lr.predict(x)

# print("Valor de BrAC (x): {}".format(x))
# print("Valor predito de BAC (y^): {}".format(y_hat))
# print("Valor real de BAC (Y real):{}".format(y))
# print("Valor do residuo: {}".format(y-y_hat[0]))

y_val_hat = lr.predict(x_val)
# print("Valores de X: {}".format(x_val))
# print("Valores da predição de Y: {}".format(y_val_hat))
# print("Valores de y real: {}".format(y_val))

x_teste = 0.90
y_teste_hat = lr.predict(x_teste)
# print("Para o valor {} foi previsto o valor {}".format(x_teste, y_teste_hat))

intercept = lr.intercept_
slop = lr.coef_

# print("Intercept(ALPHA) -> {}\n Slop(BETA) -> {}".format(intercept, slop))
# print("Residuo => {}".format(((y_hat-y_treino)**2).sum()))

plt.scatter(x_treino, y_treino, color='blue')
plt.plot(x_treino, lr.predict(x_treino), color='black', linewidth=3)
plt.xlabel('Respiração (BrAC)')
plt.ylabel('Sangue (BAC)')
plt.title('Taxa de eliminação de alcool - FEMININO')
plt.show()
