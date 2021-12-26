import csv
import matplotlib.pyplot as plt
import networkx as nx

def data():
    with open('tubedata.csv') as csv_file:
        dummy_nx_graph = nx.Graph()
        tube_mapping = csv.reader(csv_file, skipinitialspace=True)
        for row in tube_mapping:
            print(row)
            dummy_nx_graph.add_edge(row[0], row[1], weight=float(row[3]))

    return dummy_nx_graph


# plt.show()
def show_weighted_graph(networkx_graph, node_size, font_size, fig_size):
    # Allocate the given fig_size in order to have space for each node
    plt.figure(num=None, figsize=fig_size, dpi=80)
    plt.axis('off')
    # Compute the position of each vertex in order to display it nicely
    nodes_position = nx.spring_layout(networkx_graph)
    # You can change the different layouts depending on your graph
    # Extract the weights corresponding to each edge in the graph
    edges_weights = nx.get_edge_attributes(networkx_graph, 'weight')
    # Draw the nodes (you can change the color)
    nx.draw_networkx_nodes(networkx_graph, nodes_position, node_size=node_size,
                           node_color=["pink"] * networkx_graph.number_of_nodes())
    # Draw only the edges
    nx.draw_networkx_edges(networkx_graph, nodes_position,
                           edgelist=list(networkx_graph.edges), width=2)
    # Add the weights
    nx.draw_networkx_edge_labels(networkx_graph, nodes_position,
                                 edge_labels=edges_weights)
    # Add the labels of the nodes
    nx.draw_networkx_labels(networkx_graph, nodes_position, font_size=font_size,
                            font_family='serif')
    plt.axis('off')
    plt.show()

show_weighted_graph(data(), 500, 8, (30, 30))


#BFS variation2 implementation

def bfs_implementation(graph, origin, destination, counter = 0, reverse=False):
  # Add current place to already_visited
  next_already_visited = [origin]
  # List of existent paths (for now only origin)
  total_paths = [[origin]]

  # Will perform exploration of all current paths
  while len(total_paths)!= 0:
    new_total_paths = []
    # I check every single existing path for now
    for path in total_paths:
      # Last element in path, where to go next?
      last_element_in_path = path[-1]
      # Nodes connected to here...
      nodes_found = list(reversed(list(graph.neighbors(last_element_in_path)))) if reverse else list(graph.neighbors(last_element_in_path))
      # Found destination!
      if destination in nodes_found:
        # Result complete, will return this path with destination at end
        return path + [destination], counter+1

      # Otherwise, I'll need to explore the nodes connected to here...
      for node in nodes_found:
        # I only will consider nodes not visited before (avoid loops and going back)
        if node not in next_already_visited:
          counter += 1
          # this node will be out of limits for next explorations
          next_already_visited.append(node)
          # I add this possible path for further exploration
          new_total_paths = new_total_paths + [path + [node]]
    # At the end, I need to explore only these "new" paths, until I reach destination, or run out of possible valid paths
    total_paths = new_total_paths

  # If no more possible paths, means solution does not exist
  return [],-1


bfs_path, _ = bfs_implementation(data(), 'Euston', 'Victoria')
print(bfs_path)
number_visited_bfs = bfs_implementation(data(), 'Euston', 'Victoria')
print("Number of visited for BFS variatin 2: {}".format(number_visited_bfs))

def compute_path_cost(graph, path):
  """
    Compute cost of a path
  """
  cost = 0
  for index_city in range(len(path) - 1):
    cost += graph[path[index_city]][path[index_city + 1]]["weight"]
  return cost

bfs_cost_path = compute_path_cost(data(), bfs_path)

print("Cost for BFS path {}: {}".format(bfs_path, bfs_cost_path))
