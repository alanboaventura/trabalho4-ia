def criar_linha(b0, b1, vetor_x):
    linhaRegressao = []

    for i in range(len(vetor_x)):
        y = b0 + (b1 * vetor_x[i])
        linhaRegressao += [y]

    return linhaRegressao