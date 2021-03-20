import numpy as np


def bird_function(x, y):
    return np.sin(x) * np.e ** (1 - np.cos(y)) ** 2 + np.cos(y) * np.e ** (1 - np.sin(x)) ** 2 + (x - y) ** 2


class Individuo:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def reproduce(pais, n_filhos, taxa_mutacao, taxa_crossover):
    inherits1 = np.random.random(n_filhos)
    inherits2 = np.random.random(n_filhos)
    crossoverx = np.random.random(n_filhos)
    crossovery = np.random.random(n_filhos)
    mutationsx = np.random.random(n_filhos)
    mutationsy = np.random.random(n_filhos)
    xs = np.zeros(n_filhos)
    ys = np.zeros(n_filhos)
    filhos = list()
    for idx in range(n_filhos):
        mutationsx[idx] = np.random.random(1) / 2 + 0.75 if mutationsx[idx] > 1 - taxa_mutacao else 1
        mutationsy[idx] = np.random.random(1) / 2 + 0.75 if mutationsy[idx] > 1 - taxa_mutacao else 1
        inherits1[idx] = (np.abs(np.linspace(0, 1, len(pais)) - inherits1[idx])).argmin()
        inherits2[idx] = (np.abs(np.linspace(0, 1, len(pais)) - inherits2[idx])).argmin()
        if crossoverx[idx] < (1-taxa_crossover)/2:
            crossoverx[idx] = 0
        elif crossoverx[idx] < (1-taxa_crossover):
            crossoverx[idx] = 1
        else:
            crossoverx[idx] = 2
        if crossovery[idx] < (1-taxa_crossover)/2:
            crossovery[idx] = 0
        elif crossovery[idx] < (1-taxa_crossover):
            crossovery[idx] = 1
        else:
            crossovery[idx] = 2
    inherits1 = inherits1.astype(int)
    inherits2 = inherits2.astype(int)
    crossoverx = crossoverx.astype(int)
    crossovery = crossovery.astype(int)
    for idx in range(n_filhos):
        if crossoverx[idx] == 0:
            xs[idx] = pais[inherits1[idx]].x
        elif crossoverx[idx] == 1:
            xs[idx] = pais[inherits2[idx]].x
        else:
            xs[idx] = (pais[inherits2[idx]].x + pais[inherits1[idx]].x) / 2
        xs[idx] *= mutationsx[idx]
        if crossovery[idx] == 0:
            ys[idx] = pais[inherits1[idx]].y
        elif crossovery[idx] == 1:
            ys[idx] = pais[inherits2[idx]].y
        else:
            ys[idx] = (pais[inherits2[idx]].y + pais[inherits1[idx]].y) / 2
        ys[idx] *= mutationsy[idx]
        filhos.append(Individuo(xs[idx], ys[idx]))
    return filhos


def evaluate(individuos):
    results = list()
    for individuo in individuos:
        results.append(bird_function(individuo.x, individuo.y))
    results = np.array(results).reshape(-1)
    sorter = results.argsort()
    results.sort()
    individuos = [individuos[s] for s in sorter]
    return results, individuos


generations = 10
n_individuos = 100
n_selecionados_por_geracao = 10
individuos = [Individuo(np.random.random(1) * 10, np.random.random(1) * 10) for i in range(n_individuos)]

for gen in range(generations):
    print("Geração ", gen)
    results, individuos = evaluate(individuos)
    individuos = reproduce(individuos[:n_selecionados_por_geracao], n_individuos, 1/3, 0.05)
    print("Mínimo obtido na geração: ", results[0])
    print("Coordenadas do melhor indivíduo: ", individuos[0].x, individuos[0].y)
