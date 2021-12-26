import csv
import matplotlib.pyplot as plt
import networkx as nx

def tube_data():
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

show_weighted_graph(tube_data(), 500, 8, (30, 30))

#DFS implementation-variation 1

# Your function implementing DFS
def construct_path_from_root(node, root):
    """the non-recursive way!"""

    path_from_root = [node['label']]
    while node['parent']:
        node = node['parent']
        path_from_root = [node['label']] + path_from_root
    return path_from_root


def my_depth_first_graph_search(nxobject, initial, goal, compute_exploration_cost=False, reverse=False):
    """the no-oop way!"""

    frontier = [{'label': initial, 'parent': None}]
    explored = {initial}
    number_of_explored_nodes = 1

    while frontier:
        node = frontier.pop()  # pop from the right of the list
        number_of_explored_nodes += 1
        if node['label'] == goal:
            if compute_exploration_cost:
                print('number of explorations = {}'.format(number_of_explored_nodes))
            return node

        neighbours = reversed(list(nxobject.neighbors(node['label']))) if reverse else nxobject.neighbors(node['label'])
        for child_label in neighbours:

            child = {'label': child_label, 'parent': node}
            if child_label not in explored:
                frontier.append(child)  # added to the right of the list, so it is a LIFO
                explored.add(child_label)
    return None


solution = my_depth_first_graph_search(tube_data(), 'Euston', 'Victoria')
construct_path_from_root(solution, 'Euston')
solution_dfs = my_depth_first_graph_search(tube_data(), 'Euston', 'Victoria', True)

#DFS implementation-variation 2

def my_recursive_dfs_implementation(graph, origin, destination, already_visited=[], count=1, reverse=False):
    """
      Recursive way to implement DFS
    """
    # If I reach destination, I finish right here, return list with the final place
    if origin == destination:
        return [origin], count + 1

    next_already_visited = already_visited.copy()
    # Add current place to already_visited
    next_already_visited.append(origin)

    neighbours = reversed(list(graph.neighbors(origin))) if reverse else graph.neighbors(origin)
    # Check all possible destinations from where I am
    for next_node in neighbours:
        # Will only go if I havent gone before (No Loops nor going back)
        if next_node not in already_visited:
            # Go to first node possible
            result, count = my_recursive_dfs_implementation(graph, next_node, destination, next_already_visited, count,
                                                            reverse)
            # If not dead end, means I found. Otherwise try next node
            if result != []:
                path = [origin] + result
                return path, count + 1

    # If no node works, I return empty string which would mean dead end
    return [], count + 1

path, _ = my_recursive_dfs_implementation(tube_data(), 'Euston', 'Victoria')
print(path)
number_visited_dfs = my_recursive_dfs_implementation(tube_data(), 'Euston', 'Victoria')
print("Number of visited for DFS variation2: {}".format(number_visited_dfs))


def compute_path_cost(graph, path):
  """
    Compute cost of a path
  """
  cost = 0
  for index_city in range(len(path) - 1):
    cost += graph[path[index_city]][path[index_city + 1]]["weight"]
  return cost

dfs_cost_path = compute_path_cost(tube_data(), path)

print("Cost for DFS path {}: {}".format(path, dfs_cost_path))



