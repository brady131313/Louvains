import random
from operator import itemgetter
import time
import math

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm

class Graph:
    def __init__(self):
        self.adjacency = {}
        self.nodeCount = 0
        self.edgeCount = 0
        self.totalWeight = 0
    
    def fromFile(self, filename, offset):
        with open(f"samples/{filename}") as file:
            next(file)
            next(file)

            for line in file:
                temp = line.rsplit()
                v1, v2 = int(temp[0]) - offset, int(temp[1]) - offset
                self.addVertex(v1)
                self.addVertex(v2)
                self.addEdge(v1, v2, 1)
        self.edgeCount = int(self.edgeCount / 2)
        self.totalWeight /= 2
        return self

    def fromMapping(self, graph, mapping):
        adjacency = {}
        mapping = self._shiftVertcies(mapping)

        for v, community in enumerate(mapping):
            self.addVertex(community)
            if community in adjacency:
                adjacency[community].append(v)
            else:
                adjacency[community] = [v]
        
        for v in adjacency:
            vertciesInV = adjacency[v]
            adjacency[v] = {}
            for u in vertciesInV:
                neighbors = graph.neighbors(u)
                communities = [(mapping[e[0]], e[1]) for e in neighbors]
                for c in communities:
                    if c[0] in adjacency[v]:
                        adjacency[v][c[0]] += c[1]
                    else:
                        adjacency[v][c[0]] = c[1]
        
        for v in adjacency:
            for u, w in adjacency[v].items():
                self.addEdge(v, u, w)

        return [self, mapping]
    
    def _shiftVertcies(self, mapping):
        offsetMapping = {}
        offset = 0

        for i in range(len(mapping)):
            if mapping[i] not in offsetMapping:
                offsetMapping[mapping[i]] = offset
                offset += 1
        for i in range(len(mapping)):
            mapping[i] = offsetMapping[mapping[i]]
        
        return mapping

    def addVertex(self, v):
        if v not in self.adjacency:
            self.adjacency[v] = set()
            self.nodeCount += 1

    def addEdge(self, v1, v2, w = 1):
        if v1 not in self.adjacency or v2 not in self.adjacency:
            raise Exception("Vertex not in graph")

        self.adjacency[v1].add((v2, w))
        self.adjacency[v2].add((v1, w))
        self.edgeCount += 1
        self.totalWeight += w

    def neighbors(self, v):
        if v not in self.adjacency:
            raise Exception("Vertex not in graph")

        return self.adjacency[v]

    def isAdjacent(self, v1, v2):
        if v1 in self.adjacency and v2 in self.adjacency:
            neighbors = [e[0] for e in self.neighbors(v1)]
            return v2 in neighbors
        return False

    def weightedSelfLoops(self, v):
        if v not in self.adjacency:
            raise Exception("Vertex not in graph")
        
        for edge in self.neighbors(v):
            if edge[0] == v:
                return edge[1]
        
        return 0

    def weightedDegree(self, v):
        if v not in self.adjacency:
            raise Exception("Vertex not in graph")

        return sum([e[1] for e in self.adjacency[v]])

    def __str__(self):
        str = ""
        for v in sorted(self.adjacency):
            edges = [e[0] for e in self.neighbors(v)]
            #edges = self.neighbors(v)
            str += f"{v} -> {edges}\n"
        str += f"Nodes: {self.nodeCount}, edges: {self.edgeCount}, weight: {self.totalWeight}"
        return str

class Community:
    def __init__(self, graph):
        self.graph = graph
        self.nodeToCommunity = [0] * graph.nodeCount 
        self.weightsInCommunity = [0] * graph.nodeCount
        self.weightsIncident = [0] * graph.nodeCount

        for i in range(graph.nodeCount):
            self.nodeToCommunity[i] = i
            self.weightsInCommunity[i] = graph.weightedSelfLoops(i)
            self.weightsIncident[i] = graph.weightedDegree(i)

    def _insert(self, vertex, community, degreeVertexToCommunity):
        self.weightsIncident[community] += self.graph.weightedDegree(vertex)
        self.weightsInCommunity[community] += 2 * degreeVertexToCommunity + self.graph.weightedSelfLoops(vertex)
        self.nodeToCommunity[vertex] = community

    def _remove(self, vertex, community, degreeVertexToCommunity):
        self.weightsIncident[community] -= self.graph.weightedDegree(vertex)
        self.weightsInCommunity[community] -= 2 * degreeVertexToCommunity + self.graph.weightedSelfLoops(vertex)
        self.nodeToCommunity[vertex] = -1
        
    def modularity(self):
        m2 = self.graph.totalWeight
        q = 0

        for inC, totC in zip(self.weightsInCommunity, self.weightsIncident):
            if totC > 0:
                q += (inC / m2) - ((totC / m2) * (totC / m2))
        return q

    def modularityGain(self, vertex, community, degreeVertexToCommunity):
        weightsIncidient = self.weightsIncident[community]
        degreeInC = self.graph.weightedDegree(vertex)

        return (degreeVertexToCommunity - weightsIncidient * degreeInC / self.graph.totalWeight)

    def neighborsCommunities(self, vertex):
        neighbors = [e[0] for e in self.graph.neighbors(vertex)]
        weights = [e[1] for e in self.graph.neighbors(vertex)]

        neighborsCommunities = {self.nodeToCommunity[neighbor]: 0 for neighbor in neighbors}
        neighborsCommunities[self.nodeToCommunity[vertex]] = 0
        
        for neighbor, weight in zip(neighbors, weights):
            neighborsCommunities[self.nodeToCommunity[neighbor]] += weight
        
        return neighborsCommunities

    def iteration(self):
        Q = self.modularity()
        QOld = Q
        print(Q)

        while True:
            QOld = Q
            for v in self.graph.adjacency:
                initialCommunity = self.nodeToCommunity[v]
                communityMap = self.neighborsCommunities(v)

                self._remove(v, initialCommunity, communityMap[initialCommunity])
                dQs = [(com, self.modularityGain(v, com, deg)) for com, deg, in communityMap.items()]
                
                maxDq = max(dQs, key = itemgetter(1))
                self._insert(v, maxDq[0], communityMap[maxDq[0]])

            Q = self.modularity()
            if abs(Q - QOld) == 0:
                break
        return [Q, self.nodeToCommunity]
        
def findCommunities(graph):
    Q = -1
    G = graph
    vertexMap = [i for i in range(graph.nodeCount)]
    while True:
        QOld = Q

        C = Community(G)
        Q, mapping = C.iteration()
        G, mapping = Graph().fromMapping(G, mapping)
        vertexMap = [mapping[i] for i in vertexMap]

        if abs(Q - QOld) == 0:
            break
    return vertexMap

def getColors(mappings):
    colors1 = [
        "#FFFFB300", 
        "#FF803E75", 
        "#FFFF6800", 
        "#FFA6BDD7", 
        "#FFC10020", 
        "#FFCEA262", 
        "#FF817066"
    ]

    colors2 = [
        (0, 0, 1),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (1, 0, 1),
        (1, .5, .5),
        (.5, .5, .5),
        (.5, 0, 0),
        (1, .5, 0)
    ]

    mappings = [colors2[i % len(colors2)] for i in mappings]
    return mappings

def visualizeGraph(graph, mappings):
    G = nx.Graph()
    edges = []

    for v1 in graph.adjacency:
        neighbors = [e[0] for e in graph.neighbors(v1)]
        weights = [e[1] for e in graph.neighbors(v1)]
        for neighbor, weight in zip(neighbors, weights):
            edges.append((v1, neighbor, weight))

    colors = getColors(mappings)
    #cmap = cm.get_cmap('viridis')
    #colors = cmap(colors)

    G.add_weighted_edges_from(edges)
    layout = nx.spring_layout(G)

    nx.draw_networkx(G, pos=layout, node_color=colors, node_size=200, alpha=0.75, with_labels=False)
    plt.show()

def main():
    graph = Graph().fromFile("karate.txt", 0)
    mappings = findCommunities(graph)

    maxCluster = max(mappings)
    print(len(set(mappings)))
    
    '''
    communities = set(mappings)
    for c in communities:
        vertcies = list(graph.adjacency.keys())
        cluster = filter(lambda x: mappings[x] == c, vertcies)
        print(f"Cluster: {c} = {list(cluster)}")
    '''
    
    visualizeGraph(graph, mappings)


if __name__ == "__main__":
    main()

        
