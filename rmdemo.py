# Integrantes: Alan Boaventura e Lucas Carvalho
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
# Dependendo da IDE utilizada, no nosso caso PyCharm, pode ser indicado que a importação do mplot3d não é utiliza.
# Porém ele é utilizado, e caso for removido irá causar erro na execução do código.
from mpl_toolkits.mplot3d import Axes3D

def regmultipla(x, y):
    # Equação para os parâmetros do modelo
    # 𝛽= (Xt X)-¹ Xt y

    X_values = x[:, (0, 1, 2)]
    # Xt X
    matriz_vezes_transposta = np.dot(X_values.T, X_values)
    # (Xt X)-¹
    matriz_invertida = np.linalg.inv(matriz_vezes_transposta)
    # (Xt X)-¹ Xt
    matriz_invertida_vezes_transposta = np.dot(matriz_invertida, X_values.T)
    # (Xt X)-¹ Xt y
    matriz_invertida_vezes_transposta_vezes_y = np.dot(matriz_invertida_vezes_transposta, y)

    return matriz_invertida_vezes_transposta_vezes_y

# Essa função será responsável por calcular a correlação de dois vetores recebidos por parâmetro.
# O código está preparado para trabalhar com vetores Nx1, mas é necessário que ambos tenham o mesmo valor de N, ou seja,
# o mesmo número de colunas
def correlacao(vetor_x, vetor_y):
    # Equação de correlação utilizada nessa função
    # r = Σ(x−x̄)(y−ȳ) / √(Σ(x−x̄)² Σ(y−ȳ)²)
    # Para facilitar a visualização, a equação está divida em 3 variáveis
    # dividendo = Σ(x−x̄)(y−ȳ)
    # divisor1 = Σ(x−x̄)²
    # divisor2 = Σ(y−ȳ)

    # Calcula a média dos valores dos vetores através da função mean
    media_x = statistics.mean(vetor_x)
    media_y = statistics.mean(vetor_y)

    dividendo = 0

    # Realiza o calculo do dividendo
    for i in range(len(vetor_x)):
        dividendo += ((vetor_x[i] - media_x) * (vetor_y[i] - media_y))

    divisor1 = 0
    divisor2 = 0

    # Realiza o calculo do divisor 1 e divisor 2
    for i in range(len(vetor_x)):
        # Para realizar as operações de potenciação (n²) e radiciação (√n) não serão utilizadas funções preparadas
        # A potenciação será realizada pelo operador n ** n
        # A radiciação será realizada pelo operador n ** (1/2)
        divisor1 += ((vetor_x[i] - media_x) ** 2)
        divisor2 += ((vetor_y[i] - media_y) ** 2)

    divisor_final = (divisor1 * divisor2) ** (1 / 2)

    return round(dividendo / divisor_final, 4)


def regressao(vetor_x, vetor_y):
    # Equação de regressão
    # y = β0 + β1x
    # Onde:
    # 𝛽0 = 𝑦̄ − β1𝑥̄
    # 𝛽1 = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²

    # Calcula a média dos valores dos vetores através da função mean
    media_x = statistics.mean(vetor_x)
    media_y = statistics.mean(vetor_y)

    dividendo = 0
    divisor = 0

    # Realiza o calculo do dividendo
    for i in range(len(vetor_x)):
        dividendo += ((vetor_x[i] - media_x) * (vetor_y[i] - media_y))
        divisor += ((vetor_x[i] - media_x) ** 2)

    b1 = dividendo / divisor
    b0 = media_y - b1 * media_x

    return round(b0, 4), round(b1, 4)


# Gera uma lista com a linha de regressão
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

# Separa dos dados utilizados o seu respectivo X e y
X = data.values[:, [0, 1, 2]]
y = data.values[:, [3]]

x_tamanho = X[:, 1]
x_quartos = X[:, 2]
y_preco = y[:, 0]

# Calcula a correlação entre tamanho x preço e nr. quartos x preço
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
