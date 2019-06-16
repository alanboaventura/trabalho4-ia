# Integrantes: Alan Boaventura e Lucas Carvalho
import matplotlib.pyplot as plt
import statistics

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


# Esse método irá criar a linha de regressão
def criar_linha(b0, b1, vetor_x):
    linha_regressao = []

    for i in range(len(vetor_x)):
        y = b0 + (b1 * vetor_x[i])
        linha_regressao += [y]

    return linha_regressao


def gerar_grafico(x, y, b0, b1, r):
      plt.scatter(x, y)
      plt.plot(x, criar_linha(b0, b1, x))
      plt.title("Correlação: " + str(r) + "  B0: " + str(b0) + "  B1: " + str(b1))
      plt.ylabel("Preço")
      plt.xlabel("Tamanho")

# datasets
x1 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]

x2 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y2 = [9.14, 8.14, 8.47, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]

x3 = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19]
y3 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50]

# x1 y1
r = correlacao(x1, y1)
b0, b1 = regressao(x1, y1)

gerar_grafico(x1, y1, b0, b1, r)
plt.figure()

# x2 y2
r = correlacao(x2, y2)
b0, b1 = regressao(x2, y2)

gerar_grafico(x2, y2, b0, b1, r)
plt.figure()

# x3 y3
r = correlacao(x3, y3)
b0, b1 = regressao(x3, y3)

gerar_grafico(x3, y3, b0, b1, r)
plt.show()

print("3) Qual dos datasets não é apropriado para regressão linear?")
print("R: Analisando os gráficos de dispersão é possível perceber que o dataset 2, composto por x2 e y2, cria uma "
      "linha de regressão polinomial. Portanto, não é apropriado para a regressão linear.")
