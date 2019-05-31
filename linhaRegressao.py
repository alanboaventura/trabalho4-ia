def criar_linha(b0, b1, vetor_x, vetor_y):
    max_valor = 0
    if (max(vetor_x) > max(vetor_y)):
        max_valor = int(max(vetor_x)) + 5
    else: max_valor = int(max(vetor_y)) + 5

    linhaRegressao = []

    for i in range(max_valor):
        y = b0 + (b1 * i)
        linhaRegressao += [y]

    return linhaRegressao