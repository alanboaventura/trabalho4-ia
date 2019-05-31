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
    # divisor2 = Σ(y−ȳ)²

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

    divisorFinal = (divisor1 * divisor2) ** (1/2)

    return round(dividendo / divisorFinal, 4)