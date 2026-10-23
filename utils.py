import matplotlib.pyplot as plt
import numpy as np
import heapq
import random


def grid_canvas(xlim: int, ylim: int, points: list[int], edges: list[tuple]) -> None:
    """
    Crea el gráfico de una cuadrícula [0, x] X [0, y]
    Colorea los puntos de la lista points y aristas de la lista edges

    Los edges se reciben en formato ((x1, y1), (x2, y2))
    """
    x = [a for a, _ in points]
    y = [b for _, b in points]
    scale = 4  # Queda bonito en esta escala

    plt.scatter(x, y, color='teal', s=scale, alpha=1, edgecolors='none')

    for e in edges:
        plt.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]], color='purple')

    plt.xlim(0, xlim)
    plt.ylim(0, ylim)
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal')

    plt.show()


def dijkstra(xlim: int, ylim: int, adj: np.array, src: tuple) -> np.array:
    """
    Adaptado del código de https://www.geeksforgeeks.org
    Corre el algoritmo de dijkstra para obtener la distancia mínima 
    entre el vértice fuente src y resto de vértices.
    El array np.array guarda para cada par (a, b) una lista de elementos
    de la forma ((c, d), peso). 
    Se retorna un np.array que guarda para cada par (a, b) la distacia a src
    """
    # Min-heap (priority queue) storing pairs of (distance, node)
    pq = []

    x = xlim + 1
    y = ylim + 1
    dist = np.full((x, y), np.inf)
    ancestros = np.empty((x, y), dtype=object)
    ancestros.fill((-1, -1))

    # Distance from source to itself is 0
    dist[*src] = 0
    heapq.heappush(pq, (0, src))

    # Process the queue until all reachable vertices are finalized
    while pq:
        d, u = heapq.heappop(pq)

        # If this distance not the latest shortest one, skip it
        if d > dist[*u]:
            continue

        # Explore all neighbors of the current vertex
        for v, w in adj[*u]:

            # If we found a shorter path to v through u, update it
            if dist[*u] + w < dist[*v]:
                dist[*v] = dist[*u] + w
                heapq.heappush(pq, (dist[*v], v))
                ancestros[*v] = u

    # Return the final shortest distances from the source
    return dist, ancestros


def reconstruir_geo(inicio: tuple, final: tuple, ancestros: list[tuple]) -> list[tuple]:
    geodesica = []
    vtc = final
    geodesica.append(vtc)
    while vtc != inicio:
        vtc = ancestros[*vtc]
        geodesica.append(vtc)
    return geodesica


def fpp(xlim: int, ylim: int, gen) -> np.array:
    """
    Recibe una función func que genera números aleatorios (iid)
    y devuelve un np.array que guarda para cada vértice (a, b), 
    una lista con sus vecinos en el formato ((c, d), peso_arista)
    """
    x = xlim + 1
    y = ylim + 1
    pesos = np.empty((x, y), dtype=object)

    for i in range(0, xlim + 1):
        for j in range(0, ylim + 1):
            pesos[i, j] = []

    for i in range(0, xlim + 1):
        for j in range(0, ylim + 1):
            if i + 1 <= xlim:
                peso = gen()
                pesos[i, j].append(((i + 1, j), peso))
                pesos[i + 1, j].append(((i, j), peso))
            if j + 1 <= ylim:
                peso = gen()
                pesos[i, j].append(((i, j + 1), peso))
                pesos[i, j + 1].append(((i, j), peso))

    return pesos


if __name__ == "__main__":
    n = 20
    x = np.linspace(0, n, n + 1)
    y = x

    plt.scatter(x, y, c='purple', s=8, alpha=1, edgecolors='none')

    plt.xlim(- 0.5, n + 0.5)
    plt.ylim(- 0.5, n + 0.5)
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal')

    environment = fpp(n, n, lambda: random.expovariate(1))
    dists, ancestros = dijkstra(n, n, environment, (0, 0))

    # for a in range(7):
    #     for b in range(7):
    #         plt.text(a, b, f"{dists[a, b]:.2f}", ha='center')

    for a in range(n + 1):
        for b in range(n + 1):
            # for tupla, peso in environment[a, b]:
            #     x = (a + tupla[0]) / 2
            #     y = (b + tupla[1]) / 2
            #     plt.text(x, y, f"{peso:.2f}", ha='center')

            if a == 0 and b == 0:
                continue

            ancestro = ancestros[a, b]
            plt.plot((a, ancestro[0]), (b, ancestro[1]), color='purple', lw=3)

    plt.show()
