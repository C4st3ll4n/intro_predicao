import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset = pd.read_csv('dados/dados_masc', index_col=0)

var_ind = ['BrAC (mg/L/h)']
var_dep = 'BAC (g/L/h)'

x_treino, x_val, y_treino, y_val = train_test_split(dataset[var_ind], dataset[var_dep], test_size=0.15)

lr = LinearRegression()

lr.fit(x_treino, y_treino)

x = x_val.iloc[0]
y = y_val.iloc[0]
yChapeu = lr.predict(x)

y_val_hat = lr.predict(x_val)

x_teste = 0.082
y_teste_hat = lr.predict(x_teste)
print("Resultado da predição do valor de teste:{}".format(y_teste_hat))
intercept = lr.intercept_
slop = lr.coef_

plt.scatter(x_treino, y_treino, color='black')
plt.plot(x_treino, lr.predict(x_treino), color='red', linewidth=3)
plt.xlabel('Respiração (BrAC)')
plt.ylabel('Sangue (BAC)')
plt.title('Taxa de eliminação de alcool - MASCULINO')
plt.show()

