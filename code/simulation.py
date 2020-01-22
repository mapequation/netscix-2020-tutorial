# Copyright 2019 Anton Eriksson
# See LICENSE
from collections import namedtuple, defaultdict
from numpy import random

PhysNode = namedtuple("PhysNode", "id, name")
StateNode = namedtuple("StateNode", "id, phys_node, field, is_inter, choices, links")
Network = namedtuple("Network", "phys_nodes, state_nodes")


def example_network(num_nodes=2):
    fields = ("phys", "biol")

    phys_nodes = {i + 1: PhysNode(i + 1, f"{field} inter {i + 1}") for i, field in enumerate(fields)}

    states = {field: {} for field in fields}

    max_phys_id = max(phys_nodes.keys())
    max_state_id = 0

    for field in fields:
        state_nodes = states[field]

        state_ids = [max_state_id + i + 1 for i in range(4)]
        max_state_id += 4

        state_nodes[state_ids[0]] = StateNode(state_ids[0], phys_nodes[1], field, True, [state_ids[3]],
                                              defaultdict(int))
        state_nodes[state_ids[1]] = StateNode(state_ids[1], phys_nodes[2], field, True, [state_ids[2]],
                                              defaultdict(int))

        state_nodes[state_ids[2]] = StateNode(state_ids[2], phys_nodes[1], field, True, [], defaultdict(int))
        state_nodes[state_ids[3]] = StateNode(state_ids[3], phys_nodes[2], field, True, [], defaultdict(int))

        choices = []

        for i in range(num_nodes):
            max_phys_id += 1
            max_state_id += 1
            phys_id = max_phys_id
            state_id = max_state_id
            phys_nodes[phys_id] = PhysNode(phys_id, f"{field} {phys_id}")
            state_nodes[state_id] = StateNode(state_id, phys_nodes[phys_id], field, False, [], defaultdict(int))
            choices.append(state_id)

        for node in state_nodes.values():
            node.choices.extend(choices)
            if not node.is_inter:
                node.choices.extend((state_ids[0], state_ids[1]))

    state_nodes = {**states["phys"], **states["biol"]}

    return Network(phys_nodes, state_nodes)


def run(network, n, steps, mu, return_paths=True, self_links=False):
    state_nodes = network.state_nodes
    paths = []
    weight = 1

    start_nodes = list(node for node in state_nodes.values() if not node.is_inter)
    start_indices = random.choice(len(start_nodes), n)

    for start_index in start_indices:
        current = start_nodes[start_index]
        path = [current.phys_node.id]

        for step in range(steps):
            if current.is_inter and random.random() <= mu:
                neighbor_id = random.choice([node.id for node in state_nodes.values()
                                             if node.phys_node == current.phys_node and
                                             node is not current])
                neighbor = state_nodes[neighbor_id]
                choices = [id_ for id_ in neighbor.choices if not state_nodes[id_].is_inter]
            else:
                choices = current.choices if self_links else [id_ for id_ in current.choices if id_ is not current.id]

            next_ = state_nodes[random.choice(choices)]

            if return_paths:
                path.append(next_.phys_node.id)

            current.links[next_.id] += weight
            current = next_

        if return_paths:
            path.append(weight)
            paths.append(path)

    if return_paths:
        aggregated_str_paths = defaultdict(int)

        for path in paths:
            *steps, weight = path
            aggregated_str_paths[" ".join(map(str, steps))] += weight

        aggregated_paths = [[*map(int, path.split(" ")), weight] for path, weight in aggregated_str_paths.items()]

        aggregated_paths.sort(key=lambda path: path[-1], reverse=True)
        return aggregated_paths

    return None


def write_partitions(filename, network, partitions):
    field_to_module_id = defaultdict(lambda: len(field_to_module_id))

    def module_id(node):
        if partitions == 1:
            return 0
        elif partitions > 1:
            return field_to_module_id[node.field]

    flow = 1 / len(network.state_nodes)

    state_nodes = sorted(network.state_nodes.values(), key=lambda node: node.field)

    with open(filename, "w") as f:
        f.write("# state_id module flow physical_id\n")
        f.writelines(f"{node.id} {module_id(node)} {flow} {node.phys_node.id}\n"
                     for node in state_nodes)


def write_net(filename, network):
    phys_nodes = network.phys_nodes
    state_nodes = network.state_nodes

    with open(filename, "w") as f:
        f.write(f"*Vertices {len(phys_nodes)}\n")
        f.write("#physical_id name\n")
        f.writelines(f'{node.id} "{node.name}"\n' for node in phys_nodes.values())

        f.write("*States\n")
        f.write("#state_id physical_id\n")
        f.writelines(f"{node.id} {node.phys_node.id}\n" for node in state_nodes.values())

        f.write("*Links\n")
        f.write("#source_id target_id weight\n")
        for node in state_nodes.values():
            f.writelines(f"{node.id} {target} {weight}\n" for target, weight in node.links.items())


def write_paths(filename, network, paths):
    phys_nodes = network.phys_nodes

    with open(filename, "w") as f:
        f.write(f"*Vertices {len(phys_nodes)}\n")
        f.write("#physical_id name\n")
        f.writelines(f'{node.id} "{node.name}"\n' for node in phys_nodes.values())

        f.write("*Paths\n")
        f.writelines(f"{' '.join(map(str, path))}\n" for path in paths)
