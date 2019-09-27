import numpy as np

class DecisionTreeClassificator:
    def __init__(self, criterion):
        """
        criterion: Какой критерий расщепления использовать ("Gini" или "Entropy")
        """
        self.criterion = criterion
        self.features_and_splits_list = []
        self.leaves_and_criterions_list = []
        self.max_leaf_criterion = None
        
    def fit(self, X, y, verbose=False):
        self.leaves_and_criterions_list.append((X, y, self.criterion_count(y)))
        self.max_leaf_criterion = self.leaves_and_criterions_list[0][-1]
        k = 1
        while abs(self.max_leaf_criterion) > 1e-2:
            sum_crit = self.leaves_and_criterions_list[0][-1]
            for n, (node_X, node_y, crit) in enumerate(self.leaves_and_criterions_list):
                if crit > 0:
                    feature_number, split_value = self._find_best_split(node_X, node_y)
                    if feature_number is not None and split_value is not None:
                        self.features_and_splits_list.append([feature_number, split_value])
                        left_node_X, left_node_y, right_node_X, right_node_y = self._split_leaf(node_X, node_y, 
                                                                                                feature_number, 
                                                                                                split_value)
                     
                        del self.leaves_and_criterions_list[n]
                        self.leaves_and_criterions_list.append((left_node_X, left_node_y, self.criterion_count(left_node_y)))
                        self.leaves_and_criterions_list.append((right_node_X, right_node_y, self.criterion_count(right_node_y)))
                        current_max_crit = max(self.leaves_and_criterions_list[-1][2], 
                                               self.leaves_and_criterions_list[-2][2])
                        self.max_leaf_criterion = max(self.max_leaf_criterion, current_max_crit)
                        if verbose:
                            print('{} Iteration'.format(k))
                        k += 1
            if sum([crit for (_, _, crit) in self.leaves_and_criterions_list]) == 0:
                if verbose:
                    print('Success!')
                break

        
    def _split_leaf(self, node_X, node_y, feature_number, split_value):
        left_indexes = node_X[:, feature_number] < split_value
        right_indexes = node_X[:, feature_number] >= split_value
        left_node_X, left_node_y = node_X[left_indexes], node_y[left_indexes]
        right_node_X, right_node_y = node_X[right_indexes], node_y[right_indexes]
        return left_node_X, left_node_y, right_node_X, right_node_y

    def _find_best_split(self, node_X, node_y):
        current_best_feature_number = None
        current_best_split_value = None
        previous_best_crit_improvement = 0
        for feature_number in range(node_X.shape[1]):
            possible_splits = np.unique(node_X[:, feature_number])
            possible_splits = np.array([(i + j)/2 for i, j in zip(possible_splits, 
                                                                  possible_splits[1:])])
            for split in possible_splits:
                _, left_node_y, _, right_node_y = self._split_leaf(node_X, node_y, 
                                                                   feature_number, split)
                left_crit = self.criterion_count(left_node_y)
                right_crit = self.criterion_count(right_node_y)
                current_crit = self.criterion_count(node_y)
                current_crit_improvement = current_crit - (left_node_y.shape[0]/node_y.shape[0])*\
                                           left_crit - (right_node_y.shape[0]/node_y.shape[0])*\
                                           right_crit
                if current_crit_improvement > 0 and current_crit_improvement > previous_best_crit_improvement:
                    current_best_feature_number = feature_number
                    current_best_split_value = split
                    previous_best_crit_improvement = current_crit_improvement
        return current_best_feature_number, current_best_split_value
        
    def criterion_count(self, node_y):
        if self.criterion == 'Entropy':
            probs = np.unique(node_y, return_counts=True)[1]
            probs = probs/probs.sum()
            node_critetion = sum([-prob*np.log(prob) for prob in probs])
        elif self.criterion == 'Gini':
            probs = np.unique(node_y, return_counts=True)[1]
            probs = probs/probs.sum()
            node_critetion = sum([prob*(1 - prob) for prob in probs])
        else:
            node_critetion = 'Определите валидный критерий расщепления!'
        return node_critetion
