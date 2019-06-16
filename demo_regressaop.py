# Integrantes: Alan Boaventura e Lucas Carvalho
import scipy.io as scipy
import numpy as np
import matplotlib.pyplot as plt
import random


# Método responsável por calcular o erro quadrático médio
def erro_quadratico_medio(yO, yE):
    # EQM = (sum(residuo)) / size(y,1)
    # residuo = (y - ӯ)²
    # yO = y = valor observado
    # yE = ӯ = valor estimado
    return np.mean((yO - yE) ** 2)

# Carrega datasets do arquivo data_preg
mat = scipy.loadmat('data_preg.mat')
array_data = np.array(mat['data'])

# Define os valores de x e y, para facilitar a visualização nos próximos passos
x = array_data[:, 0]
y = array_data[:, 1]

# Calcula os valores de B (beta) para N = 1 e a linha de regressão
b = np.polyfit(x, y, 1)
r = np.polyval(b, x)
plt.plot(x, r, 'red')

# Calcula os valores de B (beta) para N = 2 e a linha de regressão
b = np.polyfit(x, y, 2)
r2 = np.polyval(b, x)
plt.plot(x, r2, 'green')

# Calcula os valores de B (beta) para N = 3 e a linha de regressão
b = np.polyfit(x, y, 3)
r3 = np.polyval(b, x)
plt.plot(x, r3, 'black')

# Calcula os valores de B (beta) para N = 8 e a linha de regressão
b = np.polyfit(x, y, 8)
r8 = np.polyval(b, x)
plt.plot(x, r8, 'yellow')

# Inserindo os dados a serem exibidos no gráfico "Original"
plt.scatter(x, y)
plt.ylabel("y")
plt.xlabel("x")
plt.title("Original")
plt.figure()

# Calcula o EQM para cada uma das linhas de regressão calculadas anteriormente
eqm1 = round(erro_quadratico_medio(y, r), 4)
eqm2 = round(erro_quadratico_medio(y, r2), 4)
eqm3 = round(erro_quadratico_medio(y, r3), 4)
eqm8 = round(erro_quadratico_medio(y, r8), 4)

print("g) Calcule o Erro Quadrático Médio (EQM) para cada linha de regressão. Qual é o mais preciso?")
print("EQM mais preciso: " + str(min(eqm1, eqm2, eqm3, eqm8)))
print("N1: " + str(eqm1))
print("N2: " + str(eqm2))
print("N3: " + str(eqm3))
print("N4: " + str(eqm8))

# Separa 10% dos dados em dados de teste
iTeste = random.sample(range(len(x)-5), int((10*len(x)) / 100))
# Os outros 90% são separados como dados de treinamento
iTreinamento = [ind for ind in range(len(x)) if ind not in iTeste]

# Inicia a lista com o número de linhas/colunas necessárias
dados_teste = np.zeros([2, len(iTeste)])
dados_treinamento = np.zeros([2, len(iTreinamento)])

# Busca em x e y os dados que serão utilizados como teste
dados_teste[0, :] = x[iTeste]
dados_teste[1, :] = y[iTeste]

# Busca em x e y os dados que serão utilizados como treinamento
dados_treinamento[0, :] = x[iTreinamento]
dados_treinamento[1, :] = y[iTreinamento]

# Calcula os valores de B (beta) para N = 1 e a linha de regressão apenas com os dados de treinamento
b1 = np.polyfit(dados_treinamento[0, :], dados_treinamento[1, :], 1)
r = np.polyval(b1, dados_treinamento[0, :])
plt.plot(dados_treinamento[0, :], r, 'red')

# Calcula os valores de B (beta) para N = 2 e a linha de regressão apenas com os dados de treinamento
b2 = np.polyfit(dados_treinamento[0, :], dados_treinamento[1, :], 2)
r2 = np.polyval(b2, dados_treinamento[0, :])
plt.plot(dados_treinamento[0, :], r2, 'green')

# Calcula os valores de B (beta) para N = 3 e a linha de regressão apenas com os dados de treinamento
b3 = np.polyfit(dados_treinamento[0, :], dados_treinamento[1, :], 3)
r3 = np.polyval(b3, dados_treinamento[0, :])
plt.plot(dados_treinamento[0, :], r3, 'black')

# Calcula os valores de B (beta) para N = 8 e a linha de regressão apenas com os dados de treinamento
b8 = np.polyfit(dados_treinamento[0, :], dados_treinamento[1, :], 8)
r8 = np.polyval(b8, dados_treinamento[0, :])
plt.plot(dados_treinamento[0, :], r8, 'yellow')

# Desenha no gráfico de Test Set os dados de treinamento em azul (cor padrão)
plt.scatter(dados_treinamento[0, :], dados_treinamento[1, :])
# Desenha no gráfico de Test Set os dados de teste em vermelho
plt.scatter(dados_teste[0, :], dados_teste[1, :], color="red")
plt.ylabel("y")
plt.xlabel("x")
plt.title("Test Set")
plt.draw()

# Utilizando a base de dados (betas) gerados pelos dados de treinamento, calcula os valores previstas para cada valor de N
r_teste = np.polyval(b1, dados_teste[0, :])
r2_teste = np.polyval(b2, dados_teste[0, :])
r3_teste = np.polyval(b3, dados_teste[0, :])
r8_teste = np.polyval(b8, dados_teste[0, :])

# Calcula o EQM usando os dados de teste
eqm1 = round(erro_quadratico_medio(dados_teste[1, :], r_teste), 4)
eqm2 = round(erro_quadratico_medio(dados_teste[1, :], r2_teste), 4)
eqm3 = round(erro_quadratico_medio(dados_teste[1, :], r3_teste), 4)
eqm8 = round(erro_quadratico_medio(dados_teste[1, :], r8_teste), 4)

print("g) (Teste Set) Calcule o Erro Quadrático Médio (EQM) para cada linha de regressão. Qual é o mais preciso?")
print("R: " + str(min(eqm1, eqm2, eqm3, eqm8)))
print("N1: " + str(eqm1))
print("N2: " + str(eqm2))
print("N3: " + str(eqm3))
print("N4: " + str(eqm8))

print("k) Que método é o mais preciso neste caso?")
print("O método test set. Por escolher uma pequena parte dos dados para calcular o erro quadrático, a possibilidade "
      "de escolher ruídos é menor. Entretanto, é possível que os dados de teste sejam ruídos, fazendo com que o "
      "erro quadrático seja pior que o original.")

plt.show()