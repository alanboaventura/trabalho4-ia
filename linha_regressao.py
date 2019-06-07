def criar_linha(b0, b1, vetor_x):
    linha_regressao = []

    for i in range(len(vetor_x)):
        y = b0 + (b1 * vetor_x[i])
        linha_regressao += [y]

    return linha_regressao
