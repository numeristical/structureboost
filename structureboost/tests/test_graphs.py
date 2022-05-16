import graphs


def test_US_graph_1():
    US_49_graph = graphs.US_48_and_DC_graph()
    assert len(US_49_graph.vertices) == 49


def test_adjacent_vertices():
    US_49_graph = graphs.US_48_and_DC_graph()
    assert US_49_graph.adjacent_vertices('TN') == {'AL', 'AR', 'GA', 'KY',
                                                   'MO', 'MS', 'NC', 'VA'}


def test_adjacent_edges():
    US_49_graph = graphs.US_48_and_DC_graph()
    assert US_49_graph.adjacent_edges('NV') == {frozenset({'ID', 'NV'}),
                                                frozenset({'NV', 'UT'}),
                                                frozenset({'AZ', 'NV'}),
                                                frozenset({'NV', 'OR'}),
                                                frozenset({'CA', 'NV'})}


def test_is_connected_1():
    US_49_graph = graphs.US_48_and_DC_graph()
    assert graphs.is_connected(US_49_graph)


def test_is_connected_2():
    US_49_graph = graphs.US_48_and_DC_graph()
    NY_deleted_graph = US_49_graph.delete_vertex('NY')
    assert not graphs.is_connected(NY_deleted_graph)


def test_num_cc_2():
    US_49_graph = graphs.US_48_and_DC_graph()
    NY_deleted_graph = US_49_graph.delete_vertex('NY')
    NY_NH_deleted_graph = NY_deleted_graph.delete_vertex('NH')
    assert graphs.num_connected_comp(NY_NH_deleted_graph) == 3


def test_uniform_random_spanning_tree():
    US_49_graph = graphs.US_48_and_DC_graph()
    span_tree, drd = US_49_graph.get_uniform_random_spanning_tree()
    assertion = (len(span_tree.vertices) == 49) and (
                 len(span_tree.edges) == 48) and (
                 graphs.is_connected(span_tree))
    assert assertion


def test_random_spanning_tree():
    US_49_graph = graphs.US_48_and_DC_graph()
    span_tree = US_49_graph.get_random_spanning_tree()
    assertion = (len(span_tree.vertices) == 49) and (
                 len(span_tree.edges) == 48) and (
                 graphs.is_connected(span_tree))
    assert assertion
