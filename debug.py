import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Especifique o caminho para o arquivo "iris.data"
file_path = 'iris.data'

# Defina os nomes das colunas (características)
column_names = ['sepal_length', 'sepal_width',
                'petal_length', 'petal_width', 'class']

# Carregue os dados do arquivo
iris_data = pd.read_csv(file_path, names=column_names, delimiter=',')

# Mapeie as classes para rótulos numéricos (por exemplo, 0 e 1)
iris_data['class'] = iris_data['class'].map(
    {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# Selecionando duas espécies para classificação (por exemplo, Setosa e Versicolor)
selected_species = [0, 1]
selected_data = iris_data[iris_data['class'].isin(selected_species)]

# Dividindo o conjunto de dados em treinamento e teste
X = selected_data[['sepal_length', 'sepal_width']].values
y = selected_data['class'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# Implementação do Perceptron


class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        self.errors = []

        for _ in range(self.n_iterations):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


# Selecionando as três espécies para classificação (Setosa, Versicolor e Virginica)
selected_species = [0, 1, 2]
selected_data = iris_data[iris_data['class'].isin(selected_species)]

# Dividindo o conjunto de dados em treinamento e teste
X = selected_data[['sepal_length', 'sepal_width']].values
y = selected_data['class'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Treinamento de três Perceptrons, um para cada classe
perceptrons = {}
for c in selected_species:
    # Convertendo para 1 se pertencer à classe c, 0 caso contrário
    y_train_c = (y_train == c).astype(int)
    perceptron = Perceptron(learning_rate=learning_rate,
                            n_iterations=n_iterations)
    perceptron.fit(X_train, y_train_c)
    perceptrons[c] = perceptron

# Visualização do Gráfico de Dispersão
plt.figure(figsize=(8, 6))

# Plot dos pontos para cada classe e respectiva fronteira de decisão
for c, perceptron in perceptrons.items():
    plt.scatter(X[y == c][:, 0], X[y == c][:, 1], label=f'Classe {c}')

    w = perceptron.weights[1:]
    b = perceptron.weights[0]
    x_decision = np.linspace(4, 7.5, 10)
    y_decision = -(w[0] * x_decision + b) / w[1]
    plt.plot(x_decision, y_decision,
             label=f'Reta de Decisão Classe {c}', linestyle='dashed')

plt.xlabel('Comprimento da Sépala')
plt.ylabel('Largura da Sépala')
plt.title('Classificação Setosa vs Versicolor vs Virginica')
plt.legend()
plt.show()
