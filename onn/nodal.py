class DummyNode:
    name = "Dummy"

    def __init__(self, children=None, parents=None):
        self.children = children if children is not None else []
        self.parents = parents if parents is not None else []


def traverse(node_in, node_out, nodes=None, dummy_mode=False):
    # i have no idea if this is any algorithm in particular.
    nodes = nodes if nodes is not None else []

    seen_up = {}
    q = [node_out]
    while len(q) > 0:
        node = q.pop(0)
        seen_up[node] = True
        for parent in node.parents:
            q.append(parent)

    if dummy_mode:
        seen_up[node_in] = True

    nodes = []
    q = [node_in]
    while len(q) > 0:
        node = q.pop(0)
        if not seen_up[node]:
            continue
        parents_added = (parent in nodes for parent in node.parents)
        if node not in nodes and all(parents_added):
            nodes.append(node)
        for child in node.children:
            q.append(child)

    if dummy_mode:
        nodes.remove(node_in)

    return nodes


def traverse_all(nodes_in, nodes_out, nodes=None):
    all_in = DummyNode(children=nodes_in)
    all_out = DummyNode(parents=nodes_out)
    return traverse(all_in, all_out, nodes, dummy_mode=True)
