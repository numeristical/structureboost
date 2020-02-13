# cython: profile=True


import numpy as np
import random
import json

class graph_undirected(object):
    """This is a class to handle undirected graphs.  Still very much a work in progress.
    Defines a graph by a set of vertices and a set of "frozensets" representing the edges.
    """

    def __init__(self, edges, vertices=set()):
        
        ## perhaps modify to do this only if a random element is not already a frozenset
        self.edges = {frozenset(edge) for edge in edges if len(edge)>1}
        if vertices == set():
            self.vertices = set().union(*list(self.edges))
        else:
            self.vertices = set(vertices)
        self.mc_partitions = []
        self.mc_partitions_max_size = 0
        self.all_connected_sets = []
        self.vertex_to_neighbors_dict = {}
        self.vertex_to_edges_dict = {}

    def precompute_neighbors(self):
        for vertex in self.vertices:
            self.adjacent_vertices(vertex)

    def adjacent_edges(self, target_vertex):
        if target_vertex in self.vertex_to_edges_dict.keys():
            return self.vertex_to_edges_dict[target_vertex]
        else:
            adjacent_edge_set = {x for x in self.edges if target_vertex in x}
            self.vertex_to_edges_dict[target_vertex] = adjacent_edge_set
            return adjacent_edge_set

    def adjacent_vertices(self, target_vertex):
        if target_vertex in self.vertex_to_neighbors_dict.keys():
            return self.vertex_to_neighbors_dict[target_vertex]
        else:
            #neighbors_and_self = set([x for sublist in self.adjacent_edges(target_vertex) for x in sublist])
            neighbors_and_self = set().union(*list(self.adjacent_edges(target_vertex)))
            out_set = neighbors_and_self-set([target_vertex])
            self.vertex_to_neighbors_dict[target_vertex] = out_set
        return out_set

    def adjacent_vertices_to_set(self, target_vertex_set):
        templist = [list(self.adjacent_vertices(x)) for x in target_vertex_set]
        neighbors_and_self = [x for sublist in templist for x in sublist]
        return set(neighbors_and_self)-target_vertex_set
    
    def random_neighbor(self, vertex):
        neighbors = list(self.adjacent_vertices(vertex))
        rand_neighbor = neighbors[np.random.randint(len(neighbors))]
        return rand_neighbor
    

    def vertex_degree(self, target_vertex):
        return len(self.adjacent_vertices(target_vertex))

    def contract_edge(self, edge, sep_str='_'):
        return contract_edge(self, edge, sep_str)

    def contract_edges_intset(self, edge):
        return contract_edge_intset(self, edge)

    def delete_vertex(self, vertex):
        return delete_vertex(self, vertex)

    def delete_vertices(self, vertex_set):
        return delete_vertices(self, vertex_set)

    def get_induced_subgraph(self, vertex_set):
        return get_induced_subgraph(self, vertex_set)

    def return_mc_partitions(self):
        if self.mc_partitions==[]:
            self.enumerate_mc_partitions()
        return(self.mc_partitions)

    def get_loop_erased_walk(self, start_vertex, end_set):
        curr_vertex = start_vertex
        walk = []
        walk.append(curr_vertex)
        while (not(curr_vertex in end_set)):
            curr_vertex = self.random_neighbor(curr_vertex)
            walk.append(curr_vertex)
        loop_erased_walk = erase_loops_from_walk(walk)
        return loop_erased_walk

    def get_uniform_random_spanning_tree(self):
        vert_list = list(self.vertices)
        num_vertices = len(vert_list)
        start_vertex = vert_list[np.random.randint(len(vert_list))]
        included_vertices = set([start_vertex])
        edge_set = set()
        while len(included_vertices)<num_vertices:
            out_vertices = list(self.vertices - included_vertices)
            out_vertex = out_vertices[np.random.randint(len(out_vertices))]
            branch = self.get_loop_erased_walk(out_vertex, included_vertices)
            edges_to_add = {frozenset([branch[i],branch[i+1]]) for i in range(len(branch)-1)}
            edge_set = edge_set.union(edges_to_add)
            included_vertices = included_vertices.union(set(branch))
        return graph_undirected(edge_set)

    def get_random_spanning_tree(self, start_vertex=-100, max_edges=-1):
        vert_list = list(self.vertices)
        if (start_vertex==(-100)):
            start_vertex = vert_list[np.random.randint(len(vert_list))]
        branch_vert_set = set()
        used_vert_set = set()
        new_edge_set = set()
        branch_vert_set.add(start_vertex)
        used_vert_set.add(start_vertex)
        if max_edges==-1:
            max_edges=(len(vert_list)-1)
        while len(new_edge_set)<max_edges:
            next_vertex = np.random.choice(list(branch_vert_set))
            out_neighbor_set = self.adjacent_vertices(next_vertex)
            eligible_set = out_neighbor_set - used_vert_set
            if len(eligible_set)==0:
                branch_vert_set.remove(next_vertex)
                continue
            else:
                next_neighbor = np.random.choice(list(eligible_set))
                new_edge_set.add(frozenset([next_vertex,next_neighbor]))
                branch_vert_set.add(next_neighbor)
                used_vert_set.add(next_neighbor)
        return(graph_undirected(new_edge_set, used_vert_set))

    def find_leaf(self):
        vert_list = list(self.vertices)
        # print(self.vertices, self.edges)
        # print(len(self.vertices), len(self.edges))
        if len(vert_list)==1:
            print('single_vertex_graph evaluated')
            return vert_list[0]
        visited_vertices=set()
        curr_vertex = vert_list[np.random.randint(len(vert_list))]
        
        while True:
            # print('curr vertex = {}'.format(curr_vertex))
            neighbors = self.adjacent_vertices(curr_vertex)
            if len(neighbors)==1:
                # print('leaf found')
                return curr_vertex
            else:
                visited_vertices.add(curr_vertex)
                # print('neighbors and visited vertices')
                # print(neighbors)
                # print(visited_vertices)
                next_neighbor_set = neighbors-visited_vertices
                curr_vertex = np.random.choice(list(next_neighbor_set))


    def enumerate_mc_partitions(self, max_size=0, verbose=False):
        """This method will examine every connected set S of size up to max_size and
        determine whether or not the complement of the set is also connected.  If the
        complement is also connected, then the partition {S, S^C} is added to the list
        self.mc_partitions"""
        
        # Default behavior is to find all maximally coarse partitions which
        # requires searching components up to size floor(n_vertices/2)
        if max_size==0:
            max_size=int(np.floor(len(self.vertices)/2))
        
        # Initialize some variables
        # The two lists below are sets of sets by size.
        # i.e. conn_sets_with_conn_complements_by_size[5] will be a set that contains
        # the connected sets of size 5 whose complements are also connected
        conn_sets_with_conn_complements_by_size = []
        conn_sets_with_disconn_complements_by_size = []
        
        # These two contain the sizes of each entry in the above lists
        num_conn_sets_with_conn_complements_list = []
        num_conn_sets_with_disconn_complements_list = []
        
        # Initialize the list with an empty set
        conn_sets_with_conn_complements_by_size.append(set())
        conn_sets_with_disconn_complements_by_size.append(set())


        # Corner case handling
        if(len(self.vertices)<=1):
            self.mc_partitions = []
            return []
        if(len(self.vertices)==2):
            vert_list = list(self.vertices)
            set1 = set()
            set2 = set()
            set1.add(vert_list[0])
            set2.add(vert_list[1])
            self.mc_partitions = [frozenset([frozenset(set1),frozenset(set2)])]
            self.max_size = 1
            return None
        
        # The connected components of size 1 are exactly the vertices
        if verbose:
            print('Evaluating connected sets of size 1')
        for vert in self.vertices:
            if is_connected(delete_vertex(self, vert)):
                conn_sets_with_conn_complements_by_size[0].add(frozenset({vert}))
            else:
                conn_sets_with_disconn_complements_by_size[0].add(frozenset({vert}))
        num_conn_sets_with_conn_complements_list.append(len(conn_sets_with_conn_complements_by_size[0]))
        num_conn_sets_with_disconn_complements_list.append(len(conn_sets_with_disconn_complements_by_size[0]))
        if verbose:
            print('num conn sets of comp_size 1 with connected complements = {}'.format(num_conn_sets_with_conn_complements_list[0]))
            print('num conn sets of comp_size 1 with disconnected complements = {}'.format(num_conn_sets_with_disconn_complements_list[0]))
            print('Evaluating connected sets of size 2')
        conn_sets_with_conn_complements_by_size.append(set())
        conn_sets_with_disconn_complements_by_size.append(set())
        
        # The connected components of size 2 are exactly the edges
        for edge in self.edges:
            if is_connected(delete_vertices(self, edge)):
                conn_sets_with_conn_complements_by_size[1].add(edge)
            else:
                conn_sets_with_disconn_complements_by_size[1].add(edge)
        num_conn_sets_with_conn_complements_list.append(len(conn_sets_with_conn_complements_by_size[1]))
        num_conn_sets_with_disconn_complements_list.append(len(conn_sets_with_disconn_complements_by_size[1]))
        if verbose:
            print('num conn sets of comp_size 2 with connected complements = {}'.format(num_conn_sets_with_conn_complements_list[1]))
            print('num conn sets of comp_size 2 with disconnected complements = {}'.format(num_conn_sets_with_disconn_complements_list[1]))
            print('num conn sets of comp_size <=2 with connected complements = {}'.format(np.sum(num_conn_sets_with_conn_complements_list)))
            print('num conn sets of comp_size <=2 with disconnected complements = {}'.format(np.sum(num_conn_sets_with_disconn_complements_list)))

        
        for comp_size in range(3, max_size+1):
            conn_sets_with_conn_complements_by_size.append(set())
            conn_sets_with_disconn_complements_by_size.append(set())

            if verbose:
                print('Evaluating connected sets of size {}'.format(comp_size))
            base_components = conn_sets_with_conn_complements_by_size[comp_size-2].union(conn_sets_with_disconn_complements_by_size[comp_size-2])
            for base_comp in base_components:
                neighbors_to_add = self.adjacent_vertices_to_set(base_comp)
                for neighbor in neighbors_to_add:
                    new_comp = set(base_comp)
                    new_comp.add(neighbor)
                    new_comp = frozenset(new_comp)
                    if ((not new_comp in conn_sets_with_conn_complements_by_size[comp_size-1]) and (not new_comp in conn_sets_with_disconn_complements_by_size[comp_size-1])):
                        if is_connected(delete_vertices(self,new_comp)):
                            conn_sets_with_conn_complements_by_size[comp_size-1].add(new_comp)
                        else:
                            conn_sets_with_disconn_complements_by_size[comp_size-1].add(new_comp)
            num_conn_sets_with_conn_complements_list.append(len(conn_sets_with_conn_complements_by_size[comp_size-1]))
            num_conn_sets_with_disconn_complements_list.append(len(conn_sets_with_disconn_complements_by_size[comp_size-1]))
                
            if verbose:
                print('num conn set of comp_size {} with connected complements= {}'.format(comp_size,num_conn_sets_with_conn_complements_list[comp_size-1]))
                print('num conn set of comp_size {} with discconnected complements= {}'.format(comp_size,num_conn_sets_with_disconn_complements_list[comp_size-1]))
                print('num conn set of comp_size <= {} with connected complements= {}'.format(comp_size, np.sum(num_conn_sets_with_conn_complements_list)))
                print('num conn set of comp_size <= {} with disconnected complements= {}'.format(comp_size, np.sum(num_conn_sets_with_disconn_complements_list)))
             
        self.mc_partitions = list(set([frozenset([conn_set, frozenset(self.vertices - conn_set)]) for templist in conn_sets_with_conn_complements_by_size for conn_set in templist]))
        #self.mc_partitions = [[conn_set, self.vertices - conn_set] for conn_set in conn_sets_with_conn_complements]
        self.mc_partitions_max_size = max_size
        

    def save_partitions_to_file(self, file_name):
        list_of_lists = [list(x) for x in self.all_partitions]
        with open(file_name, "w") as write_file:
            json.dump(list_of_lists, write_file)

    def load_partitions_from_file(self, file_name):
        with open(file_name, "r") as read_file:
            list_of_lists = json.load(read_file)
        self.all_partitions = [frozenset(x) for x in list_of_lists]

    
    def return_contracted_partitions(self, max_size_after_contraction = 13):
        # if len(self.contracted_partitions)>0:
        #     return self.contracted_partitions
        # else:
        new_graph = graph_undirected(self.edges, self.vertices)
        while (len(new_graph.vertices)>max_size_after_contraction):
            vertex_list = list(new_graph.vertices)
            rand_vertex = vertex_list[np.random.randint(len(vertex_list))]
            rand_vertex_neighbor_list = list(new_graph.adjacent_vertices(rand_vertex))
            rand_neighbor = rand_vertex_neighbor_list[np.random.randint(len(rand_vertex_neighbor_list))]
            new_graph = new_graph.contract_edge([rand_vertex, rand_neighbor],sep_str='_|_')
        
        new_graph.enumerate_mc_partitions()
        self.contracted_partitions = transform_partition_list(new_graph.mc_partitions, sep='_|_')
        return self.contracted_partitions

    def return_contracted_partitions_intset(self, max_size_after_contraction = 13):
        #cdef set new_vertices, new_edges
        #cdef int rand_vertex, rand_neighbor
        #cdef list rand_vertex_neighbor_list, reassembled_partition_list

        new_vertices = {frozenset([vertex]) for vertex in self.vertices}
        new_edges = {frozenset([frozenset([list(edge)[0]]), frozenset([list(edge)[1]])]) for edge in self.edges}
        new_graph = graph_undirected(new_edges, new_vertices)
        while (len(new_graph.vertices)>max_size_after_contraction):
            # vertex_list = list(new_graph.vertices)
            # rand_vertex = vertex_list[np.random.randint(len(vertex_list))]
            rand_vertex = random.choice(tuple(new_graph.vertices))
            # rand_vertex_neighbor_list = list(new_graph.adjacent_vertices(rand_vertex))
            # rand_neighbor = rand_vertex_neighbor_list[np.random.randint(len(rand_vertex_neighbor_list))]
            rand_neighbor = random.choice(tuple(new_graph.adjacent_vertices(rand_vertex)))
            new_graph = contract_edge_intset(new_graph,[rand_vertex, rand_neighbor])

        new_graph.enumerate_mc_partitions()
        reassembled_partition_list = transform_partition_list_intset(new_graph.mc_partitions)
        return reassembled_partition_list



    def enumerate_connected_sets(self, max_size=-1, verbose=False):
        if self.all_connected_sets:
            return self.all_connected_sets
        if(len(self.vertices)==0):
            return []
        if((len(self.vertices)>=1) and (len(self.vertices)<=2)):
            return [frozenset([list(self.vertices)[0]])]
        if max_size==(-1):
            max_size=len(self.vertices)
        connected_sets = []
        connected_sets.append(set())
        num_connected_sets_list = []
        if verbose:
            print('Evaluating components of size 1')
        for vert in self.vertices:
            connected_sets[0].add(frozenset({vert}))
        num_connected_sets_list.append(len(connected_sets[0]))
        if verbose:
            print('num connected sets of size 1 = {}'.format(num_connected_sets_list[0]))
            print('Evaluating components of size 2')
        connected_sets.append(set())
        for edge in self.edges:
            connected_sets[1].add(edge)
        num_connected_sets_list.append(len(connected_sets[1]))
        if verbose:
            print('num_connected_sets of size 2 = {}'.format(num_connected_sets_list[1]))
            print('num_connected_sets of size<=2 is {}'.format(np.sum(num_connected_sets_list)))

        
        for comp_size in range(3, max_size+1):
            connected_sets.append(set())

            if verbose:
                print('Evaluating components of size {}'.format(comp_size))
            base_components = connected_sets[comp_size-2]
            for base_comp in base_components:
                neighbors_to_add = self.adjacent_vertices_to_set(base_comp)
                for neighbor in neighbors_to_add:
                    new_comp = set(base_comp)
                    new_comp.add(neighbor)
                    new_comp = frozenset(new_comp)
                    connected_sets[comp_size-1].add(new_comp)
            num_connected_sets_list.append(len(connected_sets[comp_size-1]))
            # if memory_save:
            #     good_partitions[comp_size-2]=set()
            #     failed_partitions[comp_size-2]=set()
                
            if verbose:
                print('num_connected_sets of size {} = {}'.format(comp_size,num_connected_sets_list[comp_size-1]))
                print('num_total_connected_sets of size<={} is {}'.format(comp_size, np.sum(num_connected_sets_list)))
             
        connected_sets = [k for templist in connected_sets for k in templist]
        self.all_connected_sets = connected_sets
        return connected_sets
    
    def save_connected_sets_to_file(self, file_name):
        list_of_lists = [list(x) for x in self.all_connected_sets]
        with open(file_name, "w") as write_file:
            json.dump(list_of_lists, write_file)

    def load_connected_sets_from_file(self, file_name):
        with open(file_name, "r") as read_file:
            list_of_lists = json.load(read_file)
        self.all_connected_sets = [frozenset(x) for x in list_of_lists]

    def get_partitions_from_connected_sets(self, verbose=False, verbose_freq=1000):
        part_list = []
        conn_set_list = self.all_connected_sets.copy()
        conn_set_set = set(self.all_connected_sets)
        if verbose:
            print('checking {} connected sets'.format(len(conn_set_list)))
        for i,conn_set in enumerate(conn_set_list):
            if len(conn_set) > (len(self.vertices)/2):
                break
            complement_set = frozenset(self.vertices - conn_set)
            if complement_set in conn_set_set:
                part_list.append(conn_set)
                conn_set_list.remove(complement_set)
            if ((((i+1) % verbose_freq)) ==0):
                if verbose:
                    print('Checked {} sets'.format(i+1))
                    print('Found {} partitions'.format(len(part_list)))
        self.all_partitions = part_list


def contract_edge(graph, edge, sep_str='_|_'):
    edge_alph = list(edge)
    edge_alph.sort()
    contracted_vertex = sep_str.join((edge_alph))
    #new_vertices = (set(graph.vertices) - set(edge)).union({contracted_vertex})
    new_edges = [[contracted_vertex if y==edge_alph[0] or y==edge_alph[1] else y for y in this_edge] 
                 if edge_alph[0] in this_edge  or edge_alph[1] in this_edge else this_edge for this_edge in graph.edges]
    return graph_undirected(new_edges)

def contract_edge_intset(graph, edge):
    edge_alph = list(edge)
    processed_edge = process_edge(edge)
    
    contracted_vertex = frozenset({j for element in processed_edge for j in element })
    new_edges = [[contracted_vertex if y==edge_alph[0] or y==edge_alph[1] else y for y in this_edge] 
                 if edge_alph[0] in this_edge  or edge_alph[1] in this_edge else this_edge for this_edge in graph.edges]
    return graph_undirected(new_edges)


def process_edge(edge):
    return frozenset({i if type(i) == frozenset else frozenset([i]) for i in edge })

def delete_vertex(graph, vertex):
    new_edges = {edge for edge in graph.edges if vertex not in edge}
    new_vertices = graph.vertices - {vertex}
    return graph_undirected(new_edges, new_vertices)

def delete_vertices(graph, vertex_set):
    cdef set new_edges, new_vertices
    new_edges = {edge for edge in graph.edges if not vertex_set.intersection(edge)}
    new_vertices = graph.vertices - vertex_set
    return graph_undirected(new_edges, new_vertices)

def get_induced_subgraph(graph, vertex_set):
    vertex_set = set(vertex_set)
    new_edges = set([edge for edge in graph.edges if edge <= vertex_set])
    new_vertices = vertex_set
    new_graph = graph_undirected(new_edges, new_vertices)
    new_graph.all_connected_sets = [x for x in graph.all_connected_sets if new_vertices.issuperset(x)]
    return new_graph

def get_induced_subgraph_intset(graph, vertex_set):
    vertex_set = {frozenset([vertex]) for vertex in vertex_set}
    new_edges = set([edge for edge in graph.edges if edge <= vertex_set])
    new_vertices = vertex_set
    new_graph = graph_undirected(new_edges, new_vertices)
    new_graph.all_connected_sets = [x for x in graph.all_connected_sets if new_vertices.issuperset(x)]
    return new_graph


def is_connected(graph):
    cdef set visited_vertices, unexplored_vertices, new_vertices
    initial_vertex = next(iter(graph.vertices))
    # visited_vertices = [initial_vertex]
    visited_vertices = set([initial_vertex])
    unexplored_vertices = graph.adjacent_vertices(initial_vertex)
    while unexplored_vertices:
        curr_vertex = unexplored_vertices.pop()
        visited_vertices.add(curr_vertex)
        new_vertices = graph.adjacent_vertices(curr_vertex)
        unexplored_vertices = unexplored_vertices.union(new_vertices) - visited_vertices
    return (visited_vertices == graph.vertices)

def num_connected_comp(graph):
    initial_vertex = list(graph.vertices)[0]
    visited_vertices = [initial_vertex]
    unexplored_vertices = list(graph.adjacent_vertices(initial_vertex))
    while unexplored_vertices:
        curr_vertex = unexplored_vertices.pop(0)
        visited_vertices.append(curr_vertex)
        new_vertices = graph.adjacent_vertices(curr_vertex)
        unexplored_vertices = list(set(unexplored_vertices).union(new_vertices) - set(visited_vertices))
    if len(set(visited_vertices)) == len(set(graph.vertices)):
        return 1
    else:
        remainder_vertices = list(set(graph.vertices)-set(visited_vertices))
        remainder_edges = [edge for edge in graph.edges if edge.issubset(set(remainder_vertices))]
        return 1 + num_connected_comp(graph_undirected(remainder_edges, remainder_vertices))
    
def connected_comp_list(graph):
    initial_vertex = list(graph.vertices)[0]
    visited_vertices = [initial_vertex]
    unexplored_vertices = list(graph.adjacent_vertices(initial_vertex))
    while unexplored_vertices:
        curr_vertex = unexplored_vertices.pop(0)
        visited_vertices.append(curr_vertex)
        new_vertices = graph.adjacent_vertices(curr_vertex)
        unexplored_vertices = list(set(unexplored_vertices).union(new_vertices) - set(visited_vertices))
    if len(set(visited_vertices)) == len(set(graph.vertices)):
        return [graph]
    else:
        cc_vertices = set(visited_vertices)
        cc_edges = [edge for edge in graph.edges if edge.issubset(set(visited_vertices))]
        cc_graph = graph_undirected(cc_edges, cc_vertices)
        remainder_vertices = list(set(graph.vertices)-set(visited_vertices))
        remainder_edges = [edge for edge in graph.edges if edge.issubset(set(remainder_vertices))]
        return [cc_graph] + connected_comp_list(graph_undirected(remainder_edges, remainder_vertices))
    
def get_all_distances_from_vertex(graph, start_vertex):
    vertex_path_dist_dict=set()
    vertex_path_dist_dict[start_vertex] = 0
    unexplored_vertices = list(graph.adjacent_vertices(start_vertex))
    for vert in unexplored_vertices:
        vertex_path_dist_dict[vert]=1
    visited_vertices = [start_vertex]
    
    while unexplored_vertices and (len(vertex_path_dist_dict.keys())<len(graph.vertices)):
        curr_vertex = unexplored_vertices.pop()
        curr_dist = vertex_path_dist_dict[curr_vertex]
        visited_vertices.append(curr_vertex)
        curr_neighbors = graph.adjacent_vertices(curr_vertex)
        new_vertices = curr_neighbors - set(vertex_path_dist_dict.keys())
        for vert in new_vertices:
            vertex_path_dist_dict[vert]=curr_dist+1
        unexplored_vertices = list(new_vertices) + unexplored_vertices
    return vertex_path_dist_dict

def transform_partition_list(part_list, sep='_|_'):
    new_part_list = []
    for partition in part_list:
        new_fs_list = [[],[]]
        for i,fs in enumerate(partition):
            for item_str in fs:
                new_fs_list[i] = new_fs_list[i] + item_str.split(sep)
        new_fs = frozenset([frozenset(new_fs_list[0]), frozenset(new_fs_list[1])])
        new_part_list.append(new_fs)
    return(new_part_list)

def transform_partition_list_intset(part_list):
    out_list = [(frozenset().union(*list(part)[0]), frozenset().union(*list(part)[1])) for part in part_list]
    return(out_list)

def get_vertex_int_mapping(graph):
    vertex_list = np.sort(list(graph.vertices))
    out_dict_int = {vertex_list[i]:i for i in range(len(vertex_list))}
    return(out_dict_int)

def integerize_graph_from_dict(graph, mapping_dict):
    new_vertices = {mapping_dict[i] for i in graph.vertices}
    new_edges = {frozenset([mapping_dict[i] for i in edge]) for edge in graph.edges}
    return(graph_undirected(new_edges, new_vertices))

def integerize_graph(graph):
    out_dict_int = get_vertex_int_mapping(graph)
    new_graph = integerize_graph_from_dict(graph, out_dict_int)
    return(new_graph,  out_dict_int)

def get_all_pairwise_distances(graph):
    vertices_sorted = np.sort(list(graph.vertices))
    num_vertices = len(vertices_sorted)
    name_to_index_dict = {vertices_sorted[i]:i for i in range(num_vertices)}
    dist_mat = np.full((num_vertices, num_vertices),np.inf)
    for i in range(num_vertices):
        dist_mat[i,i]=0
    for edge in graph.edges:
        edge_as_list = list(edge)
        dist_mat[name_to_index_dict[edge_as_list[0]], name_to_index_dict[edge_as_list[1]]] = 1
        dist_mat[name_to_index_dict[edge_as_list[1]], name_to_index_dict[edge_as_list[0]]] = 1
    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                if (dist_mat[i,j] > (dist_mat[i,k] + dist_mat[k,j])):
                    dist_mat[i,j] = dist_mat[i,k] + dist_mat[k,j]
                    dist_mat[j,i] = dist_mat[i,j]
    return dist_mat, vertices_sorted

def get_shuffle_dictionary(my_list):
    out_dict = {}
    int_map = np.arange(len(my_list))
    np.random.shuffle(int_map)
    for i in range(len(my_list)):
        out_dict[my_list[i]]=my_list[int_map[i]]
    return(out_dict)

def shuffle_graph(my_graph, shuffle_dict=None):
    if shuffle_dict is None:
        shuffle_dict = get_shuffle_dictionary(list(my_graph.vertices))
    new_edge_list = []
    for edge in list(my_graph.edges):
        edge = list(edge)
        new_edge = [shuffle_dict[edge[i]] for i in range(len(edge))]
        new_edge_list.append(new_edge)
    return(graph_undirected(new_edge_list, my_graph.vertices.copy()))

def erase_loops_from_walk(walk):
    while len(walk)>len(set(walk)):
        duplicate_found=False
        i=0
        visited = {}
        while not(duplicate_found):
            duplicate_found = walk[i] in visited.keys()
            if not(duplicate_found):
                visited[walk[i]] = i
                i+=1
            else:
                walk = walk[:visited[walk[i]]]+walk[i:]
    return(walk)


def separate_by_two_vertices(graph, vert_1, vert_2):
    dict_1 = get_all_distances_from_vertex(graph, vert_1)
    dict_2 = get_all_distances_from_vertex(graph, vert_2)
    comp_2 = set([vert for vert in dict_2.keys() if dict_2[vert]<dict_1[vert]])
    comp_1 = graph.vertices - comp_2
    return comp_1, comp_2

def cycle_int_graph(range_low, range_high):
    edge_list = [[i,i+1] for i in range(range_low,range_high)]
    edge_list.append([range_high, range_low])
    return(graph_undirected(edge_list))


def US_48_and_DC_graph():
    US_graph_edges_list = [['AL','FL'],['AL','GA'],['AL','MS'],['AL','TN'],
              ['AR','LA'],['AR','MS'],['AR','TN'],['AR','MO'],['AR','OK'],['AR','TX'],
              ['AZ','CA'],['AZ','NM'],['AZ','NV'],['AZ','UT'],
              ['CA','OR'],['CA','NV'],
              ['CO','WY'],['CO','NE'],['CO','KS'],['CO','UT'],['CO','NM'],['CO','OK'],
              ['CT','RI'],['CT','NY'],['CT','MA'],
              ['DC','MD'],['DC','VA'],['DE','NJ'],['DE','MD'],['DE','PA'],
              ['FL','GA'],['GA','NC'],['GA','SC'],['GA','TN'],
              ['IA','IL'],['IA','MN'],['IA','WI'],['IA','MO'],['IA','NE'],['IA','SD'],
              ['ID','MT'],['ID','WY'],['ID','UT'],['ID','NV'],['ID','OR'],['ID','WA'],
              ['IL','KY'],['IL','MO'],['IL','IN'],['IL','WI'],
              ['IN','MI'],['IN','OH'],['IN','KY'],
              ['KS','MO'],['KS','NE'],['KS','OK'],
              ['KY','OH'],['KY','WV'],['KY','VA'],['KY','TN'],['KY','MO'],
              ['LA','MS'],['LA','TX'],['MA','VT'],['MA','NH'],['MA','NY'],['MA','RI'],
              ['MD','PA'],['MD','VA'],['MD','WV'],['ME','NH'],
              ['MI','WI'],['MI','OH'],['MN','ND'],['MN','SD'],['MN','WI'],
              ['MO','TN'],['MO','NE'],['MO','OK'],['MS','TN'],['MT','WY'],['MT','ND'],['MT','SD'],
              ['NC','SC'],['NC','VA'],['NC','TN'],['ND','SD'],
              ['NE','SD'],['NE','WY'],['NH','VT'],['NJ','NY'],['NJ','PA'],
              ['NM','TX'],['NM','OK'],['NV','OR'],['NV','UT'],['NY','PA'],['NY','VT'],
              ['OH','WV'],['OH','PA'],
              ['OK','TX'],['OR','WA'],['PA','WV'],['SD','WY'],['TN','VA'],
              ['UT','WY'],['VA','WV']]
    US_49_graph = graph_undirected(US_graph_edges_list)
    return US_49_graph

def US_50_and_DC_graph():
    """Graph of all 50 US states plus DC.  AK is considered adjacent to WA
       and HI is considered adjacent to CA."""
    US_graph_edges_list = [['AL','FL'],['AL','GA'],['AL','MS'],['AL','TN'],
              ['AR','LA'],['AR','MS'],['AR','TN'],['AR','MO'],['AR','OK'],['AR','TX'],
              ['AZ','CA'],['AZ','NM'],['AZ','NV'],['AZ','UT'],
              ['CA','OR'],['CA','NV'],
              ['CO','WY'],['CO','NE'],['CO','KS'],['CO','UT'],['CO','NM'],['CO','OK'],
              ['CT','RI'],['CT','NY'],['CT','MA'],
              ['DC','MD'],['DC','VA'],['DE','NJ'],['DE','MD'],['DE','PA'],
              ['FL','GA'],['GA','NC'],['GA','SC'],['GA','TN'],
              ['IA','IL'],['IA','MN'],['IA','WI'],['IA','MO'],['IA','NE'],['IA','SD'],
              ['ID','MT'],['ID','WY'],['ID','UT'],['ID','NV'],['ID','OR'],['ID','WA'],
              ['IL','KY'],['IL','MO'],['IL','IN'],['IL','WI'],
              ['IN','MI'],['IN','OH'],['IN','KY'],
              ['KS','MO'],['KS','NE'],['KS','OK'],
              ['KY','OH'],['KY','WV'],['KY','VA'],['KY','TN'],['KY','MO'],
              ['LA','MS'],['LA','TX'],['MA','VT'],['MA','NH'],['MA','NY'],['MA','RI'],
              ['MD','PA'],['MD','VA'],['MD','WV'],['ME','NH'],
              ['MI','WI'],['MI','OH'],['MN','ND'],['MN','SD'],['MN','WI'],
              ['MO','TN'],['MO','NE'],['MO','OK'],['MS','TN'],['MT','WY'],['MT','ND'],['MT','SD'],
              ['NC','SC'],['NC','VA'],['NC','TN'],['ND','SD'],
              ['NE','SD'],['NE','WY'],['NH','VT'],['NJ','NY'],['NJ','PA'],
              ['NM','TX'],['NM','OK'],['NV','OR'],['NV','UT'],['NY','PA'],['NY','VT'],
              ['OH','WV'],['OH','PA'],
              ['OK','TX'],['OR','WA'],['PA','WV'],['SD','WY'],['TN','VA'],
              ['UT','WY'],['VA','WV'],['HI','CA'],['AK','WA']]
    US_51_graph = graph_undirected(US_graph_edges_list)
    return US_51_graph

def CA_county_graph():
    """Graph of the 58 counties in California"""
    CA_county_graph_edges = [['San_Francisco','Alameda'],['San_Francisco','Marin'],
                    ['San_Francisco','San_Mateo'], ['San_Mateo','Alameda'],
                    ['San_Mateo','Santa_Clara'],
                    ['Alameda','Santa_Clara'],['Alameda','Contra_Costa'],
                    ['Contra_Costa','Marin'],['Marin','Sonoma'],['Napa','Sonoma'],
                    ['Napa','Solano'],['Contra_Costa','Solano'],['Santa_Cruz','Santa_Clara'],
                    ['San_Mateo','Santa_Cruz'],['Yolo','Solano'],['Napa','Yolo'],
                    ['Yolo','Sacramento'],['Contra_Costa','Sacramento'],
                    ['Sacramento','Solano'],['Sacramento','San_Joaquin'],['Alameda','San_Joaquin'],
                    ['San_Joaquin','Contra_Costa'],['San_Joaquin','Stanislaus'],
                    ['Santa_Clara','Stanislaus'],
                    ['San_Joaquin','Calaveras'],['Calaveras','Stanislaus'],['Calaveras','Amador'],
                    ['Calaveras','Alpine'],['Colusa','Yolo'],['Colusa','Glenn'],['Colusa','Butte'],
                    ['Amador','Alpine'],['El_Dorado','Alpine'],['El_Dorado','Amador'],
                    ['Sacramento','Amador'],['Sacramento','El_Dorado'],['San_Joaquin','Amador'],
                    ['Butte','Glenn'],['Monterey', 'Santa_Cruz'],['Monterey', 'Santa_Clara'],
                    ['Monterey', 'San_Benito'],['Monterey', 'Fresno'],['San_Benito', 'Santa_Clara'],
                    ['San_Benito', 'Fresno'],['San_Benito', 'Merced'],['Stanislaus', 'Merced'],
                    ['Santa_Clara', 'Merced'],['Fresno', 'Merced'],['Fresno', 'Inyo'],
                    ['Tuolumne','Alpine'],['Tuolumne','Alpine'],
                    ['Tuolumne','Calaveras'],['Tuolumne','Stanislaus'],['Tuolumne','Mariposa'],
                    ['Tuolumne','Madera'],
                    ['Tuolumne','Mono'],['Madera','Mariposa'],['Tuolumne','Merced'],
                    ['Madera','Merced'],['Madera','Mono'],['Madera','Fresno'],['Madera','Mono'],
                    ['Fresno','Mono'],
                    ['Inyo','Mono'],['Alpine','Mono'],['Monterey', 'Kings'],['Kings', 'Fresno'],
                    ['Kings', 'Tulare'],['Fresno', 'Tulare'],['Inyo', 'Tulare'],['Sutter','Yuba'],
                    ['Sutter','Placer'],
                    ['Sutter','Colusa'],['Sutter','Yolo'],['Sutter','Butte'],['Sutter','Sacramento'],
                    ['Placer','Yuba'],
                    ['Sierra','Yuba'],['Nevada','Yuba'],['Butte','Yuba'],['Sierra','Nevada'],
                    ['Butte','Sierra'],
                    ['Nevada','Placer'],['El_Dorado','Placer'],['Lake','Sonoma'],
                    ['Lake','Napa'],['Lake','Yolo'],
                    ['Lake','Colusa'],['Lake','Glenn'],
                    ['Lake','Mendocino'],['Glenn','Mendocino'],['Sonoma','Mendocino'],
                    ['San_Luis_Obispo','Monterey'],['San_Luis_Obispo','Santa_Barbara'],
                    ['San_Luis_Obispo','Kern'],
                    ['Kern','Kings'],['Kern','Tulare'],['Kern','Inyo'],
                    ['Kern','San_Bernardino'],
                    ['Kern','Los_Angeles'],['Kern','Ventura'],['Santa_Barbara','Ventura'],
                    ['Ventura','Los_Angeles'],
                    ['Los_Angeles','San_Bernardino'],['Los_Angeles','Orange'],
                    ['Orange','San_Bernardino'],['Orange','Riverside'],
                    ['Orange','San_Diego'],['Imperial','San_Diego'],['Imperial','Riverside'],
                    ['San_Bernardino','Riverside'],['Plumas','Tehama'],['Plumas','Butte'],
                    ['Plumas','Sierra'],['Plumas','Yuba'],
                    ['Plumas','Lassen'],['Plumas','Shasta'],['Butte','Tehama'],
                    ['Glenn','Tehama'],['Mendocino','Tehama'],['Trinity','Tehama'],
                    ['Shasta','Tehama'],['Shasta','Lassen'],['Shasta','Siskiyou'],
                    ['Shasta','Modoc'],['Shasta','Trinity'],
                    ['Modoc','Lassen'],['Modoc','Siskiyou'],['Trinity','Siskiyou'],
                    ['Del_Norte','Siskiyou'],['Humboldt','Siskiyou'],
                    ['Humboldt','Del_Norte'],['Humboldt','Trinity'],['Humboldt','Mendocino'],
                    ['Trinity','Mendocino']
                     ]
    CA_58_county_graph = graph_undirected(CA_county_graph_edges)
    return CA_58_county_graph
