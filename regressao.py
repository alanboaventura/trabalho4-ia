import statistics


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