import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
import random

POPULACAO = 30
GERACOES = 60
MUTACAO = 15
ELITISMO = 20
TORNEIO = 3


class INDIVIDUO:
    def __init__(self, cromossomo, vies, fitness):
        self.cromossomo = cromossomo # Pesos
        self.vies = vies
        self.fitness = fitness	 # Erro médio quadrático
        
def treina_perceptron(dadosTreino, classeTreino, pesos, learning_rate=0.01, epochs=50):
    vies = 0.0; erroQuadratico = 0.0
    for epoch in range(epochs):
        for i in range(dadosTreino.shape[0]):
            produtoEscalar = np.dot(dadosTreino[i], pesos) + vies
            
            if produtoEscalar >= 0:
                predicao = 1
            else:
                predicao = 0
             
            error = classeTreino[i] - predicao
            erroQuadratico += 0.5 * error ** 2
            pesos += learning_rate * error * dadosTreino[i]
            vies +=  error * learning_rate
    
    erroQuadratico = erroQuadratico / (epochs * dadosTreino.shape[0])
    
    return pesos, vies, erroQuadratico     

def iniciaPopulacao(dadosTreino, classeTreino):
    populacao = []
    pesos = np.random.uniform(-1, 1, dadosTreino.shape[1])
    for _ in range(POPULACAO):
        pesos, vies, erroQuadratico = treina_perceptron(dadosTreino, classeTreino, pesos)
        individuo = INDIVIDUO(pesos, vies, erroQuadratico)
        populacao.append(individuo)
    
    return populacao

def elitismo(populacao):
    aux = sorted(populacao, key=lambda x: x.fitness, reverse=	False)
    taxaDeElitismo = int(POPULACAO*ELITISMO/100)
    return aux, taxaDeElitismo

def selecaoPorTorneio(populacao):
    melhor = INDIVIDUO([], -1.0, -1.0)
    for i in range(TORNEIO):
        aux = populacao[random.randint(0, POPULACAO-1)]
        if(melhor.fitness == -1.0 or aux.fitness < melhor.fitness):
            melhor = aux
    return melhor

def recombinacaoUniforme(pai, mae):
    aux = [0 for _ in range(len(pai.cromossomo))]
    filho = INDIVIDUO(aux, -1.0, -1.0)
    
    for i in range(len(pai.cromossomo)):
        if(random.randint(0,1)):
            filho.cromossomo[i] = pai.cromossomo[i]
        else:
            filho.cromossomo[i] = mae.cromossomo[i]
    return filho

def mutacao(filho):
    aux = filho
    r = random.randint(1,100)
    
    if r <= MUTACAO:
        valor = np.random.uniform(-0.1, 0.1)
        posicao = random.randint(0,len(aux.cromossomo)-1)
        aux.cromossomo[posicao] += valor
    
    return aux

def fitness(filho, dadosTreino, classeTreino):
    aux = INDIVIDUO([],-1.0, -1.0)
    pesos, vies, erroQuadratico  = treina_perceptron(dadosTreino, classeTreino, filho.cromossomo)
    aux.cromossomo = pesos
    aux.fitness = erroQuadratico
    aux.vies = vies
    
    return aux	

def reproducao(populacao, dadosTreino, classeTreino):
	novaPopulacao = []
    
	populacao, taxaDeElitismo = elitismo(populacao)
	melhor = populacao[0]

	for i in range(POPULACAO):
		pai = selecaoPorTorneio(populacao)
		mae = selecaoPorTorneio(populacao)
  
		filho = recombinacaoUniforme(pai, mae)
		filho = mutacao(filho)
		filho = fitness(filho, dadosTreino, classeTreino)

		if filho.fitness < melhor.fitness:
			melhor = filho
   
		novaPopulacao.append(filho)
  
	i = taxaDeElitismo
	j = 0
	for i in range(taxaDeElitismo, POPULACAO):
		populacao[i] = novaPopulacao[j]
		j+=1
  
	return melhor
		            
def AG(dadosTreino, classeTreino):
    populacao = iniciaPopulacao(dadosTreino, classeTreino)
    melhor = INDIVIDUO([],0,0)
    for i in range(GERACOES):
        melhor = reproducao(populacao, dadosTreino, classeTreino)
        print(melhor.fitness)
        
    return melhor

def testa_perceptron(dadosTeste, classeTeste, pesos, vies):
	predicao = 0
	correto = 0
	valores = []
	for i in range(dadosTeste.shape[0]):
		produtoEscalar = np.dot(dadosTeste[i], pesos) + vies

		if produtoEscalar >= 0:
			predicao = 1
		else:
			predicao = 0
		if predicao == classeTeste[i]:
			correto+=1
		valores.append(predicao)
	acuracia = correto/ dadosTeste.shape[0]
	return acuracia, valores

def imprimeGrafico(dadosTestes, classeTeste, predicoes, pesosTreino, vies):
	setosa = []
	versicolour = []
	virginica = []
	for i in range(len(classeTeste)):
		if classeTeste[i] == 0:
			setosa.append(dadosTestes[i])
		elif classeTeste[i] == 1: 
			versicolour.append(dadosTestes[i])
		else: virginica.append(dadosTestes[i])
			

	setosa = np.array(setosa)
	versicolour = np.array(versicolour)
	virginica = np.array(virginica)

	plt.scatter(setosa[:,0], setosa[:,1], marker='o', label='Setosa')
	plt.scatter(versicolour[:,0], versicolour[:,1], marker='x', label='Versicolour')
	# plt.scatter(virginica[:,0], virginica[:,1], marker='s', label='Virginica')

	decisaoX = np.linspace(int(dadosTestes[:,0].min()), int(dadosTestes[:,0].max()), 100)
	#decisaoY = (-pesosTreino[0] * decisaoX - vies) / pesosTreino[1] 
	decisaoY = -(pesosTreino[0]/pesosTreino[1]*decisaoX) - (vies/pesosTreino[1])

	plt.plot(decisaoX, decisaoY, color='red', linestyle='--', label='Limite de Decisão')

	plt.xlabel('Comprimento da Sépala')
	plt.ylabel('Largura da Sépala')
	plt.title('Limite de Decisão')
	plt.legend()
	plt.show()
    
def main():
	# Importando o dataset 'iris'.
	iris = load_iris()
 
	# Parametros sepal length and sepal width
	dados = iris.data[:100, :2] 
	classes = iris.target[:100]
 
	# Separamos os dados em treino e teste.
	dadosTreino, dadosTestes, classeTreino, classeTeste = train_test_split(dados, classes, test_size=0.3, random_state=42)
	
	melhor = AG(dadosTreino, classeTreino)
	pesosTreino = melhor.cromossomo
	vies = melhor.vies 
 
	# pesos = np.random.uniform(-1, 1, dadosTreino.shape[1])
	# pesosTreino, vies, erroQuadratico = treina_perceptron(dadosTreino, classeTreino, pesos)
	# print(erroQuadratico)
	
 	# Testamos o perceptron
	acuracia, predicoes = testa_perceptron(dadosTestes, classeTeste, pesosTreino, vies)
	print(acuracia)
	imprimeGrafico(dadosTestes, classeTeste, predicoes, pesosTreino, vies)
	print("Fim")

 
 
if __name__ == "__main__":
    main()