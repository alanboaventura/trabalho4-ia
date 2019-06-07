import matplotlib.pyplot as plt
from correlacao import correlacao
from regressao import regressao
from linha_regressao import criar_linha

# datasets
x1 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]

x2 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y2 = [9.14, 8.14, 8.47, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]

x3 = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19]
y3 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50]

# x1 y1
r1 = correlacao(x1, y1)
b0, b1 = regressao(x1, y1)

plt.scatter(x1, y1)
plt.plot(x1, criar_linha(b0, b1, x1))
plt.title("Correlação: " + str(r1) + "  B0: " + str(b0) + "  B1: " + str(b1))
plt.ylabel("y")
plt.xlabel("x")
plt.figure()

# x2 y2
r1 = correlacao(x2, y2)
b0, b1 = regressao(x2, y2)

plt.scatter(x2, y2)
plt.plot(x2, criar_linha(b0, b1, x2))
plt.title("Correlação: " + str(r1) + "  B0: " + str(b0) + "  B1: " + str(b1))
plt.ylabel("y")
plt.xlabel("x")
plt.figure()

# x3 y3ada
r1 = correlacao(x3, y3)
b0, b1 = regressao(x3, y3)

plt.scatter(x3, y3)
plt.plot(x3, criar_linha(b0, b1, x3))
plt.title("Correlação: " + str(r1) + "  B0: " + str(b0) + "  B1: " + str(b1))
plt.ylabel("y")
plt.xlabel("x")
plt.show()

print("3) Qual dos datasets não é apropriado para regressão linear?")
print("R: Analisando os gráficos de dispersão é possível perceber que o dataset 2, composto por x2 e y2, cria uma "
      "linha de regressão polinomial. Portanto, não é apropriado para a regressão linear.")
