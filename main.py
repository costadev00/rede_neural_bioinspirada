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


# Parâmetros do treinamento
learning_rate = 0.1
n_iterations = 100

# Treinamento do Perceptron
perceptron = Perceptron(learning_rate=learning_rate, n_iterations=n_iterations)
perceptron.fit(X_train, y_train)

# Teste do Perceptron
y_pred = perceptron.predict(X_test)

# Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)

# Tabela de Parâmetros
parameters_table = pd.DataFrame({
    'Iterações': [n_iterations],
    'Taxa de Aprendizado': [learning_rate],
    'Conjunto de Teste': ['20% dos dados'],
    'Acurácia': [accuracy]
})

# Gráfico da Precisão
iteration_range = range(1, n_iterations + 1)
plt.plot(iteration_range, perceptron.errors, marker='o')
plt.title('Precisão do Modelo ao Longo das Iterações')
plt.xlabel('Número de Iterações')
plt.ylabel('Erros')
plt.show()

# Exibir a tabela de parâmetros
print(parameters_table)
