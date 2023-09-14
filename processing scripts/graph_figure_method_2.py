import networkx as nx
import matplotlib.pyplot as plt

# Utility function(s)
def calculate_midpoint(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2
    return [x_mid, y_mid]

# Create a factor graph using NetworkX
G = nx.Graph()

# Define variable names
variables = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'r_0', 'r_1', 'b_0', 'BLANK']

# Add variable nodes
G.add_nodes_from(variables, bipartite=0)

# Add priors
# Pose prior
G.add_node('Prior Factor', bipartite=1)
G.add_edge('Prior Factor', 'x_0')

# Buoy prior
buoy_prior_factor = 'Buoy Prior Factor'
G.add_node(buoy_prior_factor, bipartite=1)
G.add_edge(buoy_prior_factor, 'b_0')

# Rope prior(s)
# r_0
rope_prior_factor_0 = 'Rope Prior Factor 0'
G.add_node(rope_prior_factor_0, bipartite=1)
G.add_edge(rope_prior_factor_0, 'r_0')

# r_1
rope_prior_factor_1 = 'Rope Prior Factor 1'
G.add_node(rope_prior_factor_1, bipartite=1)
G.add_edge(rope_prior_factor_1, 'r_1')

# Add intermediate factor nodes and edges
pose_variables = [variable for variable in variables if variable[0] == 'x']
for i in range(len(pose_variables) - 1):
    factor_name = f'Factor {i+1}'
    G.add_node(factor_name, bipartite=1)
    G.add_edge(factor_name, variables[i])
    G.add_edge(factor_name, variables[i+1])

# Buoy detections
buoy_factor_name_1 = 'Buoy Factor 1'
G.add_node(buoy_factor_name_1, bipartite=1)
G.add_edge(buoy_factor_name_1, variables[1])
G.add_edge(buoy_factor_name_1, 'b_0')

buoy_factor_name_4 = 'Buoy Factor 3'
G.add_node(buoy_factor_name_4, bipartite=1)
G.add_edge(buoy_factor_name_4, variables[4])
G.add_edge(buoy_factor_name_4, 'b_0')

# rope detection
# r_0
rope_factor_name_2 = 'Rope Factor 2'
G.add_node(rope_factor_name_2, bipartite=1)
G.add_edge(rope_factor_name_2, variables[2])
G.add_edge(rope_factor_name_2, 'r_0')

# r_1
rope_factor_name_3 = 'Rope Factor 3'
G.add_node(rope_factor_name_3, bipartite=1)
G.add_edge(rope_factor_name_3, variables[3])
G.add_edge(rope_factor_name_3, 'r_1')

# Specify positions (x-coordinate for each node)
x_0_pos = (0, 0)
x_1_pos = (1, 0)
x_2_pos = (2, 0)
x_3_pos = (3, 0)
x_4_pos = (4, 0)
x_5_pos = (5, 0)
r_0_pos = (2, 1)
r_1_pos = (3, 1)
b_0_pos = (2.5, -1)
positions = {
    'Prior Factor': (x_0_pos[0] - 0.5, x_0_pos[1]),
    'Factor 1': calculate_midpoint(x_0_pos, x_1_pos),
    'Factor 2': calculate_midpoint(x_1_pos, x_2_pos),
    'Factor 3': calculate_midpoint(x_2_pos, x_3_pos),
    'Factor 4': calculate_midpoint(x_3_pos, x_4_pos),
    'Factor 5': calculate_midpoint(x_4_pos, x_5_pos),
    'x_0': x_0_pos,
    'x_1': x_1_pos,
    'x_2': x_2_pos,
    'x_3': x_3_pos,
    'x_4': x_4_pos,
    'x_5': x_5_pos,
    'r_0': r_0_pos,
    'r_1': r_1_pos,
    rope_prior_factor_0: (r_0_pos[0], r_0_pos[1] + 0.5),
    rope_prior_factor_1: (r_1_pos[0], r_1_pos[1] + 0.5),
    rope_factor_name_2: calculate_midpoint(x_2_pos, r_0_pos),
    rope_factor_name_3: calculate_midpoint(x_3_pos, r_1_pos),
    'b_0': b_0_pos,
    buoy_prior_factor: (b_0_pos[0], b_0_pos[1] - 0.5),
    buoy_factor_name_1: calculate_midpoint(x_1_pos, b_0_pos),
    buoy_factor_name_4: calculate_midpoint(x_4_pos, b_0_pos),
    'BLANK': (x_5_pos[0] + 0.75, 0)

}

# Define colors for nodes
prior_color = 'orange'
DR_color = 'tomato'
range_bearing_color = 'mediumseagreen'
variable_color = 'white'
rope_color = 'cornflowerblue'
rope_prior_color = 'mediumblue'
buoy_color = 'mediumpurple'
buoy_prior_color = 'darkviolet'
map_prior_color = 'teal'
node_colors = {
    'Prior Factor': prior_color,
    'Factor 1': DR_color,
    'Factor 2': DR_color,
    'Factor 3': DR_color,
    'Factor 4': DR_color,
    'Factor 5': DR_color,
    'x_0': variable_color,
    'x_1': variable_color,
    'x_2': variable_color,
    'x_3': variable_color,
    'x_4': variable_color,
    'x_5': variable_color,
    'r_0': rope_color,
    'r_1': rope_color,
    rope_prior_factor_0: rope_prior_color,
    rope_prior_factor_1: rope_prior_color,
    rope_factor_name_2: range_bearing_color,
    rope_factor_name_3: range_bearing_color,
    'b_0': buoy_color,
    buoy_prior_factor: buoy_prior_color,
    buoy_factor_name_1: range_bearing_color,
    buoy_factor_name_4: range_bearing_color,
    'BLANK': 'white'

}

node_edge_colors = {node: 'black' if node != 'BLANK' else 'white' for node in G.nodes()}

# Define sizes for nodes
factor_size = 250
variable_size = 800
node_sizes = {
    'x_0': variable_size,
    'x_1': variable_size,
    'x_2': variable_size,
    'x_3': variable_size,
    'x_4': variable_size,
    'x_5': variable_size,
    'r_0': variable_size,
    'r_1': variable_size,
    'b_0': variable_size,
    'Prior Factor': factor_size,
    'Factor 1': factor_size,
    'Factor 2': factor_size,
    'Factor 3': factor_size,
    'Factor 4': factor_size,
    'Factor 5': factor_size,
    buoy_prior_factor: factor_size,
    buoy_factor_name_1: factor_size,
    buoy_factor_name_4: factor_size,
    rope_prior_factor_0: factor_size,
    rope_prior_factor_1: factor_size,
    rope_factor_name_2: factor_size,
    rope_factor_name_3: factor_size,
    'BLANK': 50
}

# Set up plot layout with specified positions
pos = positions

# Draw the factor graph with specified node sizes and colors
nx.draw(G, pos, with_labels=False, node_color=[node_colors[node] for node in G.nodes()],
        edgecolors=[node_edge_colors[node] for node in G.nodes()], font_color='black',
        node_size=[node_sizes[node] for node in G.nodes()])

# draw only non factor labels
labels_to_use = {node: f'$\\mathit{{{node}}}$' for node in G.nodes() if node[0] in ['x', 'r', 'b'] }
nx.draw_networkx_labels(G, pos,labels=labels_to_use)

nx.draw_networkx_edges(G, pos=positions, edgelist=[('x_5', 'BLANK')], style='dashed', edge_color='black')


# Create a legend for node colors
# legend_labels = ['Prior Factor', 'Buoy Prior Factor', 'Rope Prior Factor', 'XYH Factor', 'Range/Bearing Factor']
# node_colors = [prior_color, buoy_prior_color, rope_prior_color, DR_color, range_bearing_color]
# legend_colors = node_colors
# legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for label, color in zip(legend_labels, legend_colors)]
# plt.legend(handles=legend_handles, loc='upper left')


# Show the plot
plt.show()
