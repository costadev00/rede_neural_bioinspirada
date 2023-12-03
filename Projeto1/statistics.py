import matplotlib.pyplot as plt
import pandas as pd

# Função para ler resultados de um arquivo
def read_results(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    return [int(line.split(":")[-1].strip()) for line in lines]

# Caminhos para os arquivos de resultados
dynamic_results = r"C:\Users\mathe\Documents\GitHub\rede_neural_bioinspirada\Projeto1\output\dynamic.out"
grasp_results = r"C:\Users\mathe\Documents\GitHub\rede_neural_bioinspirada\Projeto1\output\grasp.out"

# Leitura dos resultados
dynamic_values = read_results(dynamic_results)
grasp_values = read_results(grasp_results)

# Verificar se as listas têm o mesmo comprimento
if len(dynamic_values) != len(grasp_values):
    raise ValueError("As listas não têm o mesmo comprimento")

# Criar DataFrame para análise
data = {
    "Dynamic Programming": dynamic_values,
    "GRASP Algorithm": grasp_values
}

# Gerar gráfico de barras
df = pd.DataFrame(data)
df.plot(kind="bar", title="Comparação dos Algoritmos", ylabel="Valor Máximo da Mochila",xlabel="Instância", )
plt.xticks(rotation=0)
plt.legend(title="Algoritmo")
plt.show()

# Comparar estatísticas descritivas
statistics = df.describe()

# Calcular desvio padrão e média ponderada
std_deviation = df.std()
weighted_mean = (df * (1 / len(df.columns))).sum(axis=1)

# Adicionar ao DataFrame
statistics.loc['std'] = std_deviation
statistics.loc['weighted_mean'] = weighted_mean

# Gerar gráfico para o desvio padrão
std_deviation.plot(kind="bar", title="Desvio Padrão dos Algoritmos", ylabel="Desvio Padrão")
plt.xticks(rotation=0)
plt.show()

# Gerar gráfico para a média ponderada
weighted_mean.plot(kind="bar", title="Média Ponderada dos Algoritmos", ylabel="Média Ponderada")
plt.xticks(rotation=0)
plt.show()

