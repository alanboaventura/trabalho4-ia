import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

from correlacao import correlacao
from regmultipla import regmultipla
from regressao import regressao
from linha_regressao import criar_linha

# Lê o conteúdo do arquivo .csv
data = pd.read_csv("data.csv", header=None)

data.insert(0, '', 1)

X = data.values[:, [0, 1, 2]]
y = data.values[:, [3]]

r_tamanho_preco = correlacao(X[:, 1], y[:, 0])
r_quartos_preco = correlacao(X[:, 2], y[:, 0])

B = regmultipla(X, y)

# # 𝑦̂= X*𝛽
X_values = X[:, (1, 2)]
linha_regressao = np.dot(X_values, B)

b0, b1 = regressao(np.array(X[:, 1]).tolist(), np.array(y[:, 0]).tolist())

plt.scatter(X[:, 1], y)
plt.plot(np.array(X[:, 1]).tolist(), criar_linha(b0, b1, np.array(X[:, 1]).tolist()))
plt.title("Correlação: " + str(r_tamanho_preco))
plt.ylabel("Preço")
plt.xlabel("Tamanho da casa")
plt.figure()

b0, b1 = regressao(np.array(X[:, 2]).tolist(), np.array(y[:, 0]).tolist())

plt.scatter(X[:, 2], y)
plt.plot(np.array(X[:, 2]).tolist(), criar_linha(b0, b1, np.array(X[:, 2]).tolist()))
plt.title("Correlação: " + str(r_quartos_preco))
plt.ylabel("Preço")
plt.xlabel("Nr. de quartos")
plt.figure()

ax = plt.axes(projection='3d')

ax.plot3D(X[:, 1], X[:, 2], linha_regressao.flatten(), 'gray')
ax.scatter3D(X[:, 1], X[:, 2], y[:, 0])

plt.title("Correlações: \n Tamanho da casa x Preço: " + str(r_tamanho_preco) +
          " \n Número de quartos x Preço: " + str(r_quartos_preco) + "\n\n")

# 1650 e 3 quartos
X_E = np.array([1650, 3])

# 𝑦̂= X*𝛽
valor = X_E.dot(B)

print("g) Calcule o preço de uma casa que tem tamanho de 1650 e 3 quartos. O resultado deve ser igual a 293081.")
print("R: " + str(valor))

plt.show()
