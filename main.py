import numpy as np

cxb = [
 [3, 1, 3],
 [6, 5, 5],
]

bxm = [
 [100, 50],
 [50, 100],
 [50, 50]
]


# checa se a qtd de colunas da matriz 1
# iguala a qtd de linhas da matriz 2
def checarCondicaoExistencia(c1, l2):
	return c1 == l2

# extrai a coluna de uma matriz em formato de vetor
def extrairColuna(matriz, i):
	return [linha[i] for linha in matriz]

# retorna uma matriz zerada com a dimensao resultante
def dimensaoMatrizResultante(matriz1, matriz2):
	# uso a biblioteca numpy para gerar um array vazio com o tamanho necessario
	# para faciliatar os calculos
	return np.zeros(shape=(len(matriz1), len(matriz2[0])))

# calcula o produto entre duas matrizes
def calculaMatrizes(matriz1, matriz2):
	matrizResultante = dimensaoMatrizResultante(cxb, bxm)

	linhaIdx = -1
	colunaIdx = -1
	for linhaArr in matrizResultante:
		linhaIdx = linhaIdx + 1

		for valor in linhaArr:
			colunaIdx = colunaIdx + 1

			# aqui eu uso o einsum para facilitar a multiplicaçao de linha por coluna
			# https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
			linhaXcoluna = np.einsum('i,i->i', matriz1[linhaIdx], extrairColuna(matriz2, colunaIdx))

			matrizResultante[linhaIdx, colunaIdx] = linhaXcoluna.sum()

		colunaIdx = -1

	return matrizResultante


if (checarCondicaoExistencia(len(cxb[0]), len(bxm))):
	print(calculaMatrizes(cxb, bxm))
else:
	print("Condiçao de existência nao pôde ser validada!")
