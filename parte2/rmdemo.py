import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from parte1.correlacao import correlacao
from parte2.regmultipla import regmultipla
from parte1.regressao import regressao


def criar_linha(b0, b1, vetor_x):
    linha_regressao = []

    for i in range(len(vetor_x)):
        y = b0 + (b1 * vetor_x[i])
        linha_regressao += [y]

    return linha_regressao

# Lê o conteúdo do arquivo .csv
data = pd.read_csv("data.csv", header=None)

# Por padrão matemático, a matriz X deve possuir na sua primeira coluna o valor 1
# Insere na coluna 0 o valor 1
data.insert(0, '', 1)

X = data.values[:, [0, 1, 2]]
y = data.values[:, [3]]

x_tamanho = X[:, 1]
x_quartos = X[:, 2]
y_preco = y[:, 0]

r_tamanho_preco = correlacao(x_tamanho, y_preco)
r_quartos_preco = correlacao(x_quartos, y_preco)

# Chama a função para gerar o aprendizado
B = regmultipla(X, y)

# Calcula beta 0 e beta 1 da relação de tamanho da casa e preço
b0, b1 = regressao(np.array(x_tamanho).tolist(), np.array(y_preco).tolist())

plt.scatter(x_tamanho, y)
plt.plot(np.array(x_tamanho).tolist(), criar_linha(b0, b1, np.array(x_tamanho).tolist()), 'red')
plt.title("Correlação: " + str(r_tamanho_preco))
plt.ylabel("Preço")
plt.xlabel("Tamanho da casa")
plt.figure()

# Calcula beta 0 e beta 1 da relação de número de quartos e preço
b0, b1 = regressao(np.array(x_quartos).tolist(), np.array(y_preco).tolist())

plt.scatter(x_quartos, y)
plt.plot(np.array(x_quartos).tolist(), criar_linha(b0, b1, np.array(x_quartos).tolist()), 'red')
plt.title("Correlação: " + str(r_quartos_preco))
plt.ylabel("Preço")
plt.xlabel("Nr. de quartos")
plt.figure()

# Calcula a linha de regressão através da equação 𝑦̂= X*B
linha_regressao = np.dot(X, B)

ax = plt.axes(projection='3d')
ax.plot3D(x_tamanho, x_quartos, linha_regressao.flatten(), 'red')
ax.scatter3D(x_tamanho, x_quartos, y_preco)

plt.title("Correlações: \n Tamanho da casa x Preço: " + str(r_tamanho_preco) +
          " \n Número de quartos x Preço: " + str(r_quartos_preco) + "\n\n")

# 1650 e 3 quartos
X_g = np.array([1, 1650, 3])

# 𝑦̂= X*𝛽
valor = X_g.dot(B)

print("g) Calcule o preço de uma casa que tem tamanho de 1650 e 3 quartos. O resultado deve ser igual a 293081.")
print("R: " + str(valor))

plt.show()
