import numpy as np


def regmultipla(X, y):
    # Equação para os parâmetros do modelo
    # 𝛽= (Xt X)-¹ Xt y

    X_values = X[:, (1,2)]
    # Xt X
    matriz_vezes_transposta = np.dot(X_values.T, X_values)
    # (Xt X)-¹
    matriz_invertida = np.linalg.inv(matriz_vezes_transposta)
    # (Xt X)-¹ Xt
    matriz_invertida_vezes_transposta = np.dot(matriz_invertida, X_values.T)
    # (Xt X)-¹ Xt y
    matriz_invertida_vezes_transposta_vezes_y = np.dot(matriz_invertida_vezes_transposta, y)
    return matriz_invertida_vezes_transposta_vezes_y;