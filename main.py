import pandas as pd

# Especifique o caminho para o arquivo "iris.data"
file_path = 'iris.data'

# Defina os nomes das colunas (características)
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Carregue os dados do arquivo
iris_data = pd.read_csv(file_path, names=column_names, delimiter=',')

# Mapeie as classes para rótulos numéricos
iris_data['class'] = iris_data['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# Divida o conjunto de dados em treinamento e teste (por exemplo, 80% para treinamento)
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(iris_data, test_size=0.2, random_state=42)

print(iris_data)