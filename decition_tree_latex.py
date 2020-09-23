import numpy as np
from collections import Counter

def split_evaluation_metric(X, y, feature_number, threshold, criterion):
    
    left_y = y[X[:, feature_number] < threshold]
    right_y = y[X[:, feature_number] >= threshold]
    N_left = len(left_y)
    N_right = len(right_y)
    N_all = N_left + N_right
    
    if criterion == 'variance':
        
        root_metric = np.var(y)
        left_metric = np.var(left_y)
        right_metric = np.var(right_y)
        
        split_metric = root_metric - (N_left / N_all) * left_metric - (N_right / N_all) * right_metric
        print('$', f'Q = {round(root_metric, 3)} - ', '\\frac{', N_left, '}{', N_all, '}', f'\\cdot {round(left_metric, 3)} - ', '\\frac{', N_right,  '}{', N_all, '}', f'\\cdot {round(right_metric, 3)} = {round(split_metric, 3)}', r'$\\')

        
    elif criterion == 'entropy':
        
        unq_classes_dict = Counter(y)
        unq_classes_left_dict = Counter(left_y)
        unq_classes_left_dict = {**unq_classes_left_dict, 
                                 **{key: 0 for key in unq_classes_dict if key not in unq_classes_left_dict}}
        unq_classes_right_dict = Counter(right_y)
        unq_classes_right_dict = {**unq_classes_right_dict, 
                                 **{key: 0 for key in unq_classes_dict if key not in unq_classes_right_dict}}
        
        
        probs_root_dict = dict([(key, val/N_all) for key, val in unq_classes_dict.items()])
        probs_left_dict = dict([(key, val/N_left) for key, val in unq_classes_left_dict.items()])
        probs_right_dict = dict([(key, val/N_right) for key, val in unq_classes_right_dict.items()])
        
        root_metric = np.array([-prob * np.log2(prob) for prob in probs_root_dict.values() if prob != 0]).sum()
        left_metric = np.array([-prob * np.log2(prob) for prob in probs_left_dict.values() if prob != 0]).sum()
        right_metric = np.array([-prob * np.log2(prob) for prob in probs_right_dict.values() if prob != 0]).sum()
        
        split_metric = root_metric - (N_left / N_all) * left_metric - (N_right / N_all) * right_metric
        
        print('$', fr'N\_root: {N_all}, \ N\_left: {N_left}, \ N\_right: {N_right}', r'$\\')
        print()
        tmp = str({f'prob_root_{key}': round(val, 3) for key, val in probs_root_dict.items()}).replace("'", '').replace("b_r", "b\_r").replace("b_l", "b\_l").replace(", ", ", \ ")
        print('$', tmp, r'$\\')
        print()
        tmp = str({f'prob_left_{key}': round(val, 3) for key, val in probs_left_dict.items()}).replace("'", '').replace("b_r", "b\_r").replace("b_l", "b\_l").replace(", ", ", \ ")
        print('$', tmp, r'$\\')
        print()
        tmp = str({f'prob_right_{key}': round(val, 3) for key, val in probs_right_dict.items()}).replace("'", '').replace("b_r", "b\_r").replace("b_l", "b\_l").replace(", ", ", \ ")
        print('$', tmp, r'$\\')
        print()
        
        print('$', f'Q = {round(root_metric, 3)} - ', '\\frac{', N_left, '}{', N_all, '}', f'\\cdot {round(left_metric, 3)} - ', '\\frac{', N_right,  '}{', N_all, '}', f'\\cdot {round(right_metric, 3)} = {round(split_metric, 3)}', r'$\\')
        
        
    elif criterion == 'gini':
        
        unq_classes_dict = Counter(y)
        unq_classes_left_dict = Counter(left_y)
        unq_classes_left_dict = {**unq_classes_left_dict, 
                                 **{key: 0 for key in unq_classes_dict if key not in unq_classes_left_dict}}
        unq_classes_right_dict = Counter(right_y)
        unq_classes_right_dict = {**unq_classes_right_dict, 
                                 **{key: 0 for key in unq_classes_dict if key not in unq_classes_right_dict}}
        
        
        probs_root_dict = dict([(key, val/N_all) for key, val in unq_classes_dict.items()])
        probs_left_dict = dict([(key, val/N_left) for key, val in unq_classes_left_dict.items()])
        probs_right_dict = dict([(key, val/N_right) for key, val in unq_classes_right_dict.items()])
        
        root_metric = np.array([prob * (1 - prob) for prob in probs_root_dict.values() if prob != 0]).sum()
        left_metric = np.array([prob * (1 - prob) for prob in probs_left_dict.values() if prob != 0]).sum()
        right_metric = np.array([prob * (1 - prob) for prob in probs_right_dict.values() if prob != 0]).sum()
        
        split_metric = root_metric - (N_left / N_all) * left_metric - (N_right / N_all) * right_metric
        
        
        print('$', fr'N\_root: {N_all}, \ N\_left: {N_left}, \ N\_right: {N_right}', r'$\\')
        print()
        tmp = str({f'prob_root_{key}': round(val, 3) for key, val in probs_root_dict.items()}).replace("'", '').replace("b_r", "b\_r").replace("b_l", "b\_l").replace(", ", ", \ ")
        print('$', tmp, r'$\\')
        print()
        tmp = str({f'prob_left_{key}': round(val, 3) for key, val in probs_left_dict.items()}).replace("'", '').replace("b_r", "b\_r").replace("b_l", "b\_l").replace(", ", ", \ ")
        print('$', tmp, r'$\\')
        print()
        tmp = str({f'prob_right_{key}': round(val, 3) for key, val in probs_right_dict.items()}).replace("'", '').replace("b_r", "b\_r").replace("b_l", "b\_l").replace(", ", ", \ ")
        print('$', tmp, r'$\\')
        print()
        
        print('$', f'Q = {round(root_metric, 3)} - ', '\\frac{', N_left, '}{', N_all, '}', f'\\cdot {round(left_metric, 3)} - ', '\\frac{', N_right,  '}{', N_all, '}', f'\\cdot {round(right_metric, 3)} =', r'{\bf', round(split_metric, 3), '}', r'$\\')
        
    print()

    return split_metric

def find_best_split_latex(X: np.array, y: np.array, criterion='entropy'):
        
    assert criterion in ['variance', 'entropy', 'gini'], "criterion должен быть одним из ['variance', 'entropy', 'gini']"
    
    if len(set(y)) == 1:
        print('В этом листе все объекты одного класса')
    else:
    
        best_metric, best_threshold, best_feature_number = None, None, None
        for feature_number in range(X.shape[1]):

            possible_splits = np.sort(np.unique(X[:, feature_number]))
            possible_splits = np.array([(i + j) / 2 for i, j in zip(possible_splits, possible_splits[1:])])
            for threshold in possible_splits:
                print('$', r'{\bf', 'feature\\_number:', f'{feature_number + 1}, \ ', ' threshold:', round(threshold, 3), f', \ criterion: {criterion}', '}', r'$\\')
                print()
                split_metric = split_evaluation_metric(X=X, y=y, feature_number=feature_number, threshold=threshold, 
                                                       criterion=criterion)

                if best_metric is None:
                    best_metric = split_metric
                    best_threshold = threshold
                    best_feature_number = feature_number
                if best_metric < split_metric:
                    best_metric = split_metric
                    best_threshold = threshold
                    best_feature_number = feature_number
                print('\n')

            print(r'\hline')
            print('\\bigskip')

        print(r'\boxed{{\bf', fr'best\_split: [feature_{best_feature_number + 1} < {round(best_threshold, 3)}]', '}}')
            
