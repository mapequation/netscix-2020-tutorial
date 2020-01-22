# Copyright 2019 Anton Eriksson
# See LICENSE
import random
import re
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Tuple, NewType
from itertools import islice, repeat, chain

import numpy as np
from sklearn import preprocessing

from clustering_algorithm import JSdivisiveClustering

PhysicalId = NewType("PhysicalId", int)
ContainerId = NewType("ContainerId", int)
StateId = NewType("StateId", int)
Order = NewType("Order", int)


def within_context(context, iterator):
    for _, line in within_contexts((context,), iterator):
        yield line


def within_contexts(contexts, iterator):
    curr_context = None
    for line in iterator:
        if line.startswith('*'):
            l = line.lower()
            curr_context = next((context for context in contexts if l.startswith(context)), None)
            continue
        elif curr_context in contexts:
            yield curr_context, line


def window(seq, n=2):
    """Returns a sliding window (of width n) over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def repeat_first(seq, n=1):
    it = iter(seq)
    first = repeat(next(it), n)
    return chain(first, it)


class StateNode:
    __slots__ = (
        "state_id",
        "physical_id",
        "cluster_id",
        "links",
        "container",
        "out_weight",
        "out_degree",
        "is_dangling"
    )

    def __init__(self, state_id, physical_id):
        self.state_id = state_id
        self.physical_id = physical_id
        self.cluster_id = None
        self.links = {}  # state_id -> weight
        self.container = None
        self.out_weight = 0
        self.out_degree = 0
        self.is_dangling = True

    def add_link(self, target, weight):
        if self.is_dangling:
            self.is_dangling = False
        try:
            self.links[target] += weight
        except KeyError:
            self.links[target] = weight
        self.out_weight += weight
        self.out_degree += 1

    @property
    def normalized_links(self):
        return {target: weight / self.out_weight for target, weight in self.links.items()}

    @property
    def entropy_rate(self):
        return -sum(w * np.log2(w / self.out_weight) for w in self.links.values()) / self.out_weight


class StateContainer:
    container_id_counter = 0

    @classmethod
    def get_container_id(cls):
        container_id = cls.container_id_counter
        cls.container_id_counter += 1
        return container_id

    __slots__ = ("container_id", "physical_id", "parent", "children", "state_nodes", "links", "depth")

    def __init__(self, parent=None, physical_id=None, container_id=None):
        self.container_id = container_id if container_id is not None else StateContainer.get_container_id()

        if parent:
            self.physical_id = parent.physical_id
            self.depth = parent.depth + 1
            self.parent = parent
            parent.children[self.container_id] = self
        else:
            if not physical_id:
                raise RuntimeError("Need either parent or physical_id")
            self.physical_id = physical_id
            self.depth = 0
            self.parent = None

        self.children = {}  # container_id -> Container
        self.state_nodes = {}  # state_id -> StateNode
        self.links = {}  # container_id -> weight

    def __str__(self):
        if len(self.children):
            s = ""
            for child in self.children.values():
                s += f"{child}"
        else:
            s = f"<Cluster id={self.container_id}"
            if len(self.state_nodes):
                state_ids = ','.join(list(map(str, self.state_nodes.keys())))
                s += f" state_ids=[{state_ids}]"
            s += ">"
        return s

    def add_state_node(self, state_node):
        self.state_nodes[state_node.state_id] = state_node
        state_node.container = self

    def get_containers(self, depth):
        if depth == 0:
            yield self
        if depth == 1:
            yield from self.children.values()
        else:
            for child in self.children.values():
                yield from child.get_containers(depth - 1)

    @property
    def leaf_containers(self):
        if len(self.children):
            for child in self.children.values():
                yield from child.leaf_containers
        else:
            yield self

    @property
    def leaf_state_nodes(self):
        if len(self.children) > 0:
            for child in self.children.values():
                yield from child.leaf_state_nodes
        else:
            yield from self.state_nodes.values()

    @property
    def has_dangling_state_nodes(self):
        return any(state_node.is_dangling for state_node in self.state_nodes.values())

    @property
    def dangling_state_nodes(self):
        return {state_id: state_node for state_id, state_node in self.state_nodes.items()
                if state_node.is_dangling}

    @property
    def non_dangling_state_nodes(self):
        return {state_id: state_node for state_id, state_node in self.state_nodes.items()
                if not state_node.is_dangling}

    def transfer_states_to_child(self, child, state_ids):
        if child.container_id not in self.children:
            raise RuntimeError("Child not found in children")

        state_nodes = {state_id: self.state_nodes[state_id] for state_id in state_ids}
        child.state_nodes.update(state_nodes)
        for state_id, state_node in state_nodes.items():
            self.state_nodes.pop(state_id, None)
            state_node.container = child

    def transfer_states_to_children(self, label_to_state_ids, preserve_labels=False):
        for label, state_ids in label_to_state_ids.items():
            if not preserve_labels:
                label = None
            child = StateContainer(parent=self, container_id=label)
            self.transfer_states_to_child(child, state_ids)

    def transfer_child_states(self):
        if len(self.children) == 0:
            return

        for state_node in self.leaf_state_nodes:
            self.add_state_node(state_node)

        self.children.clear()


class PhysNode:
    __slots__ = ("physical_id", "name", "num_state_nodes", "container")

    def __init__(self, physical_id, name=None):
        self.physical_id = physical_id
        self.name = name
        self.num_state_nodes = 0
        self.container = StateContainer(physical_id=physical_id)

    def __str__(self):
        return f"<PhysNode id={self.physical_id} containers={self.container}>"

    def __lt__(self, other):
        return self.physical_id < other.physical_id

    def add_state_node(self, state_node):
        self.container.add_state_node(state_node)
        self.num_state_nodes += 1

    def get_containers(self, depth):
        return self.container.get_containers(depth)

    @property
    def leaf_containers(self):
        return self.container.leaf_containers

    @property
    def state_nodes(self):
        return self.container.leaf_state_nodes

    @property
    def dangling_state_nodes(self):
        return (state_node for state_node in self.state_nodes if state_node.is_dangling)

    @property
    def non_dangling_state_nodes(self):
        return (state_node for state_node in self.state_nodes if not state_node.is_dangling)


class StateNetwork:
    __slots__ = ("phys_nodes", "state_nodes", "contexts", "clusterings", "feature_matrices")

    def __init__(self):
        self.phys_nodes: Dict[PhysicalId, PhysNode] = {}
        self.state_nodes: Dict[StateId, StateNode] = {}
        self.contexts: Dict[StateId, List[List]] = {}
        self.clusterings: Dict[Order, Dict[ContainerId, JSdivisiveClustering]] = {}
        self.feature_matrices: Dict[Order, Dict[ContainerId, Tuple]] = {}

    def __str__(self):
        s = f"<StateNetwork phys_nodes=[\n"
        for phys_node in self.phys_nodes.values():
            s += f"\t{phys_node}\n"
        s += "/>"
        return s

    @property
    def num_nodes(self):
        return len(self.phys_nodes)

    @property
    def num_state_nodes(self):
        return len(self.state_nodes)

    @property
    def num_clustered_state_nodes(self):
        return len(list(self.leaf_containers))

    @property
    def num_links(self):
        return sum(state_node.out_degree for state_node in self.state_nodes.values())

    @property
    def tot_weight(self):
        return sum(node.out_weight for node in self.state_nodes.values())

    @property
    def entropy_rate(self):
        return sum(node.out_weight * node.entropy_rate for node in self.state_nodes.values()) / self.tot_weight

    @property
    def lumped_entropy_rate(self):
        containers = list(self.leaf_containers)

        h = 0

        with self.aggregated_links():
            for container in containers:
                out_weight = sum(container.links.values())
                entropy_rate = -sum(w * np.log2(w / out_weight) for w in container.links.values()) / out_weight
                h += out_weight * entropy_rate

        return h / self.tot_weight

    def get_containers(self, order):
        containers = (phys_node.get_containers(order - 2) for phys_node in self.phys_nodes.values())
        return chain(*containers)

    @property
    def leaf_containers(self):
        return chain(*(phys_node.leaf_containers for phys_node in self.phys_nodes.values()))

    def get_or_create_physical_node(self, phys_id, name=None):
        if phys_id not in self.phys_nodes:
            self.phys_nodes[phys_id] = PhysNode(phys_id, name)
        return self.phys_nodes[phys_id]

    create_physical_node = get_or_create_physical_node

    def add_state_node(self, state_node, phys_node=None):
        if state_node.state_id in self.state_nodes:
            return
        if phys_node is None:
            phys_node = self.get_or_create_physical_node(state_node.physical_id)
        phys_node.add_state_node(state_node)
        self.state_nodes[state_node.state_id] = state_node

    def add_state_link(self, source, target, weight):
        state_node = self.state_nodes[source]
        state_node.add_link(target, weight)

    def add_context(self, state_id, context):
        if state_id not in self.contexts:
            self.contexts[state_id] = []

        self.contexts[state_id].append(context)

    def context_history(self, state_id, steps=1):
        if state_id not in self.contexts:
            return

        return [context[steps] for context in self.contexts[state_id] if len(context) + 1 > steps]

    def clear_clustering(self):
        self.clusterings.clear()
        self.feature_matrices.clear()
        for phys_node in self.phys_nodes.values():
            for container in phys_node.get_containers(0):
                container.transfer_child_states()

    @classmethod
    def from_paths(cls,
                   paths_filename, output_filename=None, validation_filename=None,
                   markov_order=2,
                   create_validation=False, validation_prob=0.5, split_weight=True,
                   min_path_len=None, max_path_len=None,
                   seed=1,
                   repeat_first_entry=False, integer_weights=True):
        """Read path data and generate state network

        @param paths_filename : string, path to file with *paths data
        @param output_filename : string, path to output state network
        @param validation_filename : string, path to validation state network.
            If not None, the paths would be split into a training and a validation state network,
            keeping same state_id for state nodes with same physical n-gram,
            and non-overlapping state ids for state nodes unique to one set.
        @param markov_order : int, markov order of generated state network (default: 2)
        @param validation_prob : float, probability to save a path to the validation network
        @param split_weight : bool, treat a path with weight n as n paths of weight 1
            and save each individual path to validation network with probability validation_prob

        :returns StateNetwork
        """
        print(f"Read path data from file '{paths_filename}'...")

        np.random.seed(seed)
        num_returns = 0
        num_paths = 0
        num_ok_paths = 0
        ngram_to_state_id = defaultdict(lambda: len(ngram_to_state_id) + 1)  # ids should start with 1
        phys_names = {}
        state_network = cls()
        validation_network = cls()
        create_validation_network = create_validation or validation_filename is not None

        with open(paths_filename, 'r') as fp:
            lines = (line for line in fp if not line.startswith("#"))
            for context, line in within_contexts(("*vertices", "*paths", "*arcs"), lines):
                if context == "*vertices":
                    m = re.match(r'(\d+) "(.+)"', line)
                    if m:
                        [phys_id, phys_name] = m.groups()
                        phys_names[int(phys_id)] = phys_name
                elif context == "*paths" or context == "*arcs":
                    num_paths += 1
                    *path, weight = line.split()
                    path = [int(p) for p in path]

                    weight = int(weight) if integer_weights else float(weight)

                    path_len = len(path)
                    path_too_long = max_path_len and path_len > max_path_len
                    path_too_short = min_path_len and path_len < min_path_len
                    path_len_below_order = path_len <= markov_order and not repeat_first_entry
                    if path_len_below_order or path_too_long or path_too_short:
                        continue
                    num_ok_paths += 1

                    if path[0] == path[-1]:
                        num_returns += 1

                    weight_validation = 0
                    if create_validation_network:
                        if split_weight:
                            if integer_weights:
                                weight_validation = np.random.binomial(weight, validation_prob)
                            else:
                                weight_validation = np.random.normal(weight / 2, np.sqrt(weight / 2))
                                weight_validation = max(0, min(weight, weight_validation))
                        else:
                            weight_validation = weight if np.random.random() < validation_prob else 0
                    weight_training = weight - weight_validation
                    add_validation = weight_validation > 0
                    add_training = weight_training > 0

                    prev_state_id = None

                    if repeat_first_entry:
                        path = list(repeat_first(path, markov_order))

                    history = markov_order - 1
                    for ngram in window(path, markov_order):
                        state_id = ngram_to_state_id[ngram]

                        # Add state node
                        phys_id = ngram[-1]
                        try:
                            phys_name = phys_names[phys_id]
                        except KeyError:
                            phys_name = str(phys_id)

                        history_reversed = slice(history, -(len(path) + 1), -1)
                        history += 1

                        state_context = path[history_reversed]

                        if state_id in state_network.state_nodes:
                            state_node = state_network.state_nodes[state_id]
                        else:
                            state_node = StateNode(state_id, phys_id)
                        phys_node = state_network.get_or_create_physical_node(phys_id, phys_name)
                        state_network.add_state_node(state_node, phys_node)
                        state_network.add_context(state_id, state_context)

                        if create_validation_network:
                            if state_id in validation_network.state_nodes:
                                state_node = validation_network.state_nodes[state_id]
                            else:
                                state_node = StateNode(state_id, phys_id)
                            phys_node = validation_network.get_or_create_physical_node(phys_id, phys_name)
                            validation_network.add_state_node(state_node, phys_node)
                            validation_network.add_context(state_id, state_context)

                        if prev_state_id:
                            # Add link
                            if add_training:
                                state_network.add_state_link(prev_state_id, state_id, weight_training)
                            if add_validation:
                                validation_network.add_state_link(prev_state_id, state_id, weight_validation)

                        prev_state_id = state_id

        print(f"Done, parsed {num_ok_paths}/{num_paths} paths")
        print(f" -> {num_returns} return paths")
        print(f" -> {len(state_network.state_nodes)} states in training network")

        # print("Generated {}state network: {}".format("training " if create_validation_network else "", state_network))
        if output_filename:
            state_network.write_state_network(output_filename)

        if validation_filename:
            print(f" -> {len(validation_network.state_nodes)} states in validation network")
            # print(f"Generated validation state network: {validation_network}")
            validation_network.write_state_network(validation_filename)

        # print("Done!")
        return state_network, validation_network

    def aggregate(self, order):
        used_cluster_ids = {0}

        for container in self.get_containers(order):
            state_nodes = container.state_nodes

            steps = order - 1
            prev_ids = {state_id: self.context_history(state_id, steps) for state_id, state_node in state_nodes.items()}

            first_free_cluster_id = max(used_cluster_ids) + 1
            prev_id_to_cluster_id = defaultdict(lambda: first_free_cluster_id + len(prev_id_to_cluster_id))
            # all context numbers < (order - 1) points to the same physical node for networks sampled of that order
            context_number = 0
            cluster_ids = {state_id: prev_id_to_cluster_id[prev_id[context_number]] for state_id, prev_id in
                           prev_ids.items()}
            used_cluster_ids.update(cluster_ids.values())

            for state_id, state_node in state_nodes.items():
                state_node.cluster_id = cluster_ids[state_id]

    def get_feature_matrix(self,
                           container,
                           order,
                           normalize_rows=True,
                           normalize_matrix=False):
        try:
            # Get the container if we got a PhysNode instead of a StateContainer as first argument
            container = container.container
        except AttributeError:
            pass

        cluster_id_to_state_ids = defaultdict(list)
        for state_id, state_node in container.non_dangling_state_nodes.items():
            cluster_id_to_state_ids[state_node.cluster_id].append(state_id)

        cluster_id_to_row_index = defaultdict(lambda: len(cluster_id_to_row_index))
        target_id_to_feature_index = defaultdict(lambda: len(target_id_to_feature_index))
        row_index_to_cluster_id = {}
        dense_links = []

        for state_node in container.non_dangling_state_nodes.values():
            row_index = cluster_id_to_row_index[state_node.cluster_id]
            row_index_to_cluster_id[row_index] = state_node.cluster_id

            if order == 2:
                for target_id, weight in state_node.links.items():
                    target_physical_id = self.state_nodes[target_id].physical_id
                    feature_index = target_id_to_feature_index[target_physical_id]
                    dense_links.append((row_index, feature_index, weight))
            elif order == 3:
                for first_step_target_id, first_step_normalized_weight in state_node.normalized_links.items():
                    first_step_target_node = self.state_nodes[first_step_target_id]
                    if not first_step_target_node.is_dangling:
                        for target_id, weight in first_step_target_node.links.items():
                            target_physical_id = self.state_nodes[target_id].physical_id
                            feature_index = target_id_to_feature_index[target_physical_id]
                            dense_links.append((row_index, feature_index, first_step_normalized_weight * weight))
                    else:
                        # TODO flow model
                        # node is dangling, use first step weight
                        first_step_weight = state_node.links[first_step_target_id]
                        first_step_physical_id = first_step_target_node.physical_id
                        feature_index = target_id_to_feature_index[first_step_physical_id]
                        dense_links.append((row_index, feature_index, first_step_weight))
            else:
                raise NotImplementedError

        row_index_to_state_ids = {row_index: cluster_id_to_state_ids[cluster_id]
                                  for row_index, cluster_id in row_index_to_cluster_id.items()}

        num_rows, num_features = len(cluster_id_to_row_index), len(target_id_to_feature_index)
        feature_matrix = np.zeros((num_rows, num_features))

        if num_features == 0:
            return feature_matrix, {}

        for row_index, feature_index, weight in dense_links:
            feature_matrix[row_index][feature_index] += weight

        if normalize_rows:
            preprocessing.normalize(feature_matrix, axis=1, norm='l1', copy=False)

        if normalize_matrix:
            feature_matrix = np.divide(feature_matrix, np.sum(feature_matrix))

        return feature_matrix, row_index_to_state_ids

    def cluster_state_nodes(self,
                            order,
                            js_div_threshold,
                            merge_dangling_with_random_cluster=False,
                            merge_dangling_with_shared_context=True,
                            merge_dangling_state_nodes=False):
        if order not in self.clusterings:
            self.clusterings[order] = {}

        clustering_cache = self.clusterings[order]

        if order not in self.feature_matrices:
            self.feature_matrices[order] = {}

        feature_matrix_cache = self.feature_matrices[order]

        for container in self.get_containers(order):
            container.transfer_child_states()

            if container.container_id not in feature_matrix_cache:
                feature_matrix_cache[container.container_id] = self.get_feature_matrix(container, order)

            X, row_index_to_state_ids = feature_matrix_cache[container.container_id]

            num_states, num_features = X.shape

            if num_states < 2 or num_features < 1:
                labels = range(num_states)
            else:
                if container.container_id not in clustering_cache:
                    clustering_cache[container.container_id] = JSdivisiveClustering(X, self.tot_weight)
                divisive_clustering = clustering_cache[container.container_id]
                labels = divisive_clustering.divide_labels(js_div_threshold=js_div_threshold)

            label_to_state_ids = {label: [] for label in labels}

            for row_index, label in enumerate(labels):
                label_to_state_ids[label].extend(row_index_to_state_ids[row_index])

            # assign dangling states to clusters
            if container.has_dangling_state_nodes:
                max_label = max(label_to_state_ids.keys(), default=-1)
                first_available_label = max_label + 1

                if merge_dangling_with_random_cluster:
                    for dangling_node in container.dangling_state_nodes:
                        label_to_state_ids[random.choice(labels)].append(dangling_node)

                elif merge_dangling_with_shared_context:
                    for dangling_node in container.dangling_state_nodes:
                        dangling_node_prev_step = self.contexts[dangling_node][0][1]

                        for state_node in container.non_dangling_state_nodes:
                            found = False
                            for context in self.contexts[state_node]:
                                if context[1] == dangling_node_prev_step:
                                    for label, state_ids in label_to_state_ids.items():
                                        if state_node in state_ids:
                                            label_to_state_ids[label].append(dangling_node)
                                            found = True
                                            break
                                    break
                            if found:
                                break
                        else:
                            if len(labels) > 0:
                                label_to_state_ids[random.choice(labels)].append(dangling_node)
                            else:
                                label_to_state_ids[first_available_label + 1] = [dangling_node]
                                first_available_label += 1

                else:

                    if merge_dangling_state_nodes:
                        # add dangling nodes to a separate last cluster
                        label_to_state_ids[first_available_label] = list(container.dangling_state_nodes.keys())
                    else:
                        # add dangling nodes to their own cluster
                        for i, state_id in enumerate(container.dangling_state_nodes):
                            label_to_state_ids[first_available_label + i] = [state_id]

            container.transfer_states_to_children(label_to_state_ids)

    @contextmanager
    def aggregated_links(self):
        self.aggregate_links()
        yield
        self.clear_aggregated_links()

    def aggregate_links(self):
        for state_node in self.state_nodes.values():
            if state_node.is_dangling:
                continue

            aggregated_links = state_node.container.links

            for target_id, weight in state_node.links.items():
                target_container_id = self.state_nodes[target_id].container.container_id
                if target_container_id not in aggregated_links:
                    aggregated_links[target_container_id] = 0
                aggregated_links[target_container_id] += weight

    def clear_aggregated_links(self):
        for state_node in self.state_nodes.values():
            if not state_node.container:
                continue
            state_node.container.links.clear()

    def write_state_network(self, filename, write_contexts=False):
        print(f"Writing state network to file '{filename}'...")

        with open(filename, 'w') as f:
            f.write(f"# physical nodes: {self.num_nodes}\n")
            f.write(f"# state nodes: {self.num_state_nodes}\n")

            f.write("*Vertices\n")
            f.writelines(f'{node.physical_id} "{node.name}"\n' for node in sorted(self.phys_nodes.values()))

            f.write("*States\n")
            f.write("#state_id physical_id\n")
            f.writelines(f'{node.state_id} {node.physical_id}\n' for node in self.state_nodes.values())

            f.write("*Links\n")
            f.write("#source_id target_id weight\n")
            for source, state_node in self.state_nodes.items():
                f.writelines(f"{source} {target} {weight}\n" for target, weight in state_node.links.items())

            if write_contexts and len(self.contexts) > 0:
                f.write("*Contexts\n")
                f.write("#state_id physical_id prior_id [history...] \n")
                for state_id, contexts in self.contexts.items():
                    f.writelines(f"{state_id} {' '.join(map(str, context))}\n" for context in contexts)

    def write_clustered_network(self, filename, write_contexts=False, write_state_names=False):
        print(f"Writing clustered state network to file '{filename}'...")

        with open(filename, 'w') as f:
            containers = list(self.leaf_containers)

            f.write(f"# physical nodes: {self.num_nodes}\n")
            f.write(f"# clustered state nodes: {len(containers)}\n")

            f.write("*Vertices\n")
            f.writelines(f'{node.physical_id} "{node.name}"\n' for node in sorted(self.phys_nodes.values()))

            f.write("*States\n")
            f.write(f"#state_id physical_id {'name' if write_state_names else ''}\n")
            if write_state_names:
                f.writelines(
                    f'{c.container_id} {c.physical_id} "{",".join(list(map(str, islice(c.state_nodes.keys(), 3))))},..."\n'
                    for c in containers)
            else:
                f.writelines(f'{c.container_id} {c.physical_id}\n' for c in containers)

            f.write("*Links\n")
            f.write("#source_id target_id weight\n")
            with self.aggregated_links():
                for container in containers:
                    source = container.container_id
                    f.writelines(f"{source} {target} {weight}\n" for target, weight in container.links.items())

            if write_contexts and len(self.contexts) > 0:
                f.write("*Contexts\n")
                f.write("#state_id physical_id prior_id [history...] \n")
                for container in containers:
                    for state_node in container.state_nodes.values():
                        for context in self.contexts[state_node.state_id]:
                            f.writelines(
                                f"{container.container_id} {container.physical_id} {' '.join(map(str, context))}\n")

    def cluster_from_network(self, network):
        for phys_id, phys_node in self.phys_nodes.items():
            phys_node.container.transfer_child_states()

            unique_nodes = []
            unique_dangling_nodes = []

            label_to_state_ids = defaultdict(list)

            if phys_id not in network.phys_nodes:
                # Physical node does not exist in other network, add all state nodes to list of unique
                unique_nodes.extend(phys_node.non_dangling_state_nodes)
                unique_dangling_nodes.extend(phys_node.dangling_state_nodes)
            else:
                # Physical node exist in other network, map same state nodes to same cluster
                other_phys_node = network.phys_nodes[phys_id]
                other_state_nodes = {state_node.state_id: state_node for state_node in other_phys_node.state_nodes}

                for state_node in phys_node.state_nodes:
                    try:
                        container_id = other_state_nodes[state_node.state_id].container.container_id
                        label_to_state_ids[container_id].append(state_node.state_id)
                    except KeyError:
                        if state_node.is_dangling:
                            unique_dangling_nodes.append(state_node)
                        else:
                            unique_nodes.append(state_node)

            max_label = 0
            unique_node_labels = {}

            # Put unique state nodes in their own lumped node
            for index, state_node in enumerate(unique_nodes):
                unique_node_labels[index + max_label] = [state_node.state_id]
                max_label += 1

            # Lump dangling nodes
            if len(unique_dangling_nodes):
                unique_node_labels[max_label + 1] = [state_node.state_id for state_node in unique_dangling_nodes]

            phys_node.container.transfer_states_to_children(label_to_state_ids, preserve_labels=True)
            phys_node.container.transfer_states_to_children(unique_node_labels)
