from itertools import combinations
from typing import List, Set, FrozenSet


def get_support(x: FrozenSet[int], data_sets: List[Set[int]]) -> float:
    """
    :return: P(X)
    """
    support = 0.
    for set_i in data_sets:
        if x.issubset(set_i):
            support += 1
    return support / len(data_sets)


def get_confidence(x: FrozenSet[int], y: FrozenSet[int], data_sets: List[Set[int]]) -> float:
    """
    :return: P(Y|X)
    """
    if len(y) == 0:
        return 0.

    xy = x | y
    count_xy = 0.
    count_x = 0.
    for set_i in data_sets:
        if xy.issubset(set_i):
            count_xy += 1
        if x.issubset(set_i):
            count_x += 1
    return count_xy / count_x


def get_single_items(data_sets: List[Set[int]]) -> List[FrozenSet[int]]:
    """
    [[1,2],[2,3]] -> [[1],[2],[3]]
    """
    single_items = set()
    for set_i in data_sets:
        single_items = single_items | set_i
    ans = list()
    for i in single_items:
        ans.append(frozenset({i}))
    return ans


def in_block_list(block_list: List[FrozenSet[int]], x: FrozenSet[int]) -> bool:
    for set_i in block_list:
        if x.issuperset(set_i):
            return True
    return False


def get_candidates(base_set: List[FrozenSet[int]], set_size: int) -> Set[FrozenSet[int]]:
    """
    [[1],[2],[3]] -> [[1,2],[2,3],[1,3]] where size = 2
    [[1,2],[2,3]] -> [[1,2,3]] where size = 3
    """
    candidates = set()
    for set_i, set_j in combinations(base_set, 2):
        union = frozenset(set_i | set_j)
        if len(union) == set_size:
            candidates.add(union)
    return candidates


def get_frequent_sets(data_sets: List[Set[int]]) -> List[FrozenSet[int]]:
    set_size = 1
    block_list = list()
    frequent_sets = list()
    candidates = get_single_items(data_sets)

    while len(candidates) != 0:
        temp_list = list()
        for c in candidates:
            if in_block_list(block_list, c):
                continue
            if get_support(c, data_sets) < 0.5:
                block_list.append(c)
                continue
            temp_list.append(c)
        frequent_sets.extend(temp_list)
        set_size += 1
        candidates = get_candidates(temp_list, set_size)

    return frequent_sets


class Rule(object):
    def __init__(self, from_x: FrozenSet[int], to_y: FrozenSet[int], confidence: float):
        self.from_x = from_x
        self.to_y = to_y
        self.confidence = confidence

    def __str__(self):
        return f"{set(self.from_x)} ==> {set(self.to_y)} at: {self.confidence}"


def get_association_rules_one(set_i: FrozenSet[int], data_sets: List[Set[int]]) -> List[Rule]:
    set_size = 1
    rules = list()
    block_list = list()
    candidates = frozenset({frozenset({x}) for x in set_i})

    while len(candidates) != 0:
        temp_list = list()
        for x in candidates:
            if in_block_list(block_list, x):
                continue
            y = frozenset(set_i - x)

            confidence = get_confidence(x, y, data_sets)
            if confidence < 0.7:
                block_list.append(x)
                continue

            temp_list.append(x)
            rules.append(Rule(x, y, confidence))

        set_size += 1
        candidates = get_candidates(temp_list, set_size)

    return rules


def get_association_rules(frequent_sets: List[FrozenSet[int]], data_sets: List[Set[int]]) -> List[Rule]:
    ans = list()
    for set_i in frequent_sets:
        if len(set_i) > 1:
            ans.extend(get_association_rules_one(set_i, data_sets))
    return ans


data = [{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}]

print(get_frequent_sets(data))

for rule in get_association_rules(get_frequent_sets(data), data):
    print(rule)
