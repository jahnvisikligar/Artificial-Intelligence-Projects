from queue import PriorityQueue
import Priority_Q
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


class Node:
    """
    a class that defines the node for the nx graph.
    it consists pf city label, path cost to reach the city from the root and its parent node
    """
    def _init_(self, label, path_cost, parent):
        self.label = label
        self.path_cost = path_cost
        self.parent = parent

    def _lt_(self, other):
        return self.path_cost < other.path_cost

    def _repr_(self):
        path = construct_path_from_root(self)
        return ('(%s, %s, %s)'
                % (repr(self.label), repr(self.path_cost), repr(path)))


def construct_path(node):
    """
    this method constructs the path as a list from the root to the node
    :param node: a Node object
    :return: list
    """
    path_from_root = node['label']
    while node.parent:
        node = node.parent
        path_from_root = node['label'] + path_from_root
    return path_from_root


def construct_path_from_root(node, root):
    """the non-recursive way!"""

    path_from_root = [node['label']]
    while node['parent']:
        node = node['parent']
        path_from_root = [node['label']] + path_from_root
    return path_from_root


def remove_node_with_higher_cost(new_node, frontier):
    """
    this method removes the node from the priority queue if the new_node has a the same label but a lesser cost
    :param new_node: node
    :param frontier: priority queue
    :return: priority queue
    """
    removed = False
    frontier_list = frontier.queue
    for item in frontier_list:
        if item.label == new_node.label and item.path_cost > new_node.path_cost:
            removed_item = item
            frontier_list.remove(item)
            removed = True
            break

    if removed:
        #print("frontier = frontier - {} + {} ".format(removed_item, new_node))
        new_queue = PriorityQueue()
        frontier_list.append(new_node)
        for item in frontier_list:
            new_queue.put(item)
        return new_queue
    else:
        return frontier


def in_frontier(new_node, frontier):
    """
    this method checks if the new_node.label is already present in the frontier
    :param new_node: node
    :param frontier: priority queue
    :return: boolean
    """
    frontier_list = frontier.queue
    for item in frontier_list:
        if item.label == new_node.label:
            return True
    return False


def usc(nxobject, initial, goal,):
    """
    this method performs the uniform cost search
    :param nxobject: the weighted networkx graph
    :param initial: the initial state or root
    :param goal: the goal state or the destination
    :return: a node with the optimal path
    """
    number_of_explored_nodes = 1
    #node = Node(initial, 0, None)
    # frontier is a priority queue
    # node = {'label': initial, 'parent': None, 'weight': 0}
    # q = Priority_Q.PriorityQueue
    # # add the initial state to the priority queue
    # q.insert(node)
    # explored is a set
    # explored = set()
    #print("frontier = ", frontier.queue)
    #print("explored = ", explored)
    #
    node = {'label': initial, 'parent': None, 'weight': 0, 'line': None, 'time': 0}
    number_of_explored_nodes = 1
    # mQ = queue.PriorityQueue()
    q = Priority_Q.PriorityQueue()
    # print("Type of queue:",type(mQ))
    q.insert(node)
    explored = {initial}


    while not q.isEmpty():
        #print("\n")
        # pop the first element from the priority queue (lowest cost node)
        node = q.pop()
        #print("frontier = frontier - ", node)
        # check if the node is the goal state then return node
        if node['label'] == goal:
            print("Number of explorations = ", number_of_explored_nodes)
            return node
        # else add the node to the explored set
        number_of_explored_nodes += 1
        explored.add(node['label'])
        #print("explored = explored + ", node.label)
        # get all the neighbours of the node
        neighbours = nxobject.neighbors(node['label'])
        for child_label in neighbours:
            step_cost = nxobject.edges[(node['label'], child_label)]['weight']
            child_weight = nxobject[node['label']][child_label]['weight']
            child = {'label': child_label, 'parent': node, 'weight': child_weight}
            #child = Node(child_label, node.path_cost + step_cost, node)
            # check if the child node is already explored or not
            if child_label not in explored:
                    # and not in_frontier(child, q):
                # add the child to the frontier
                q.insert(child)
                #print("frontier = frontier + ", child)
            # if the node already exists in the frontier with a higher cost, then replace it
            # elif in_frontier(child, q):
            # q = remove_node_with_higher_cost(child, q)


solution = usc(tube_data(), 'Euston', 'Victoria')
print("Total cost: ", solution)
print("Path:", construct_path_from_root(solution, None))
#print('Number of nodes:', len(construct_path(solution)))
