from pymysql import connect
from pandas import DataFrame
from numpy import zeros, int64, int32, float64, float32, multiply, dot, identity, sum
from itertools import permutations
from math import log
import torch
import numpy as np




def setup_function(args):
    # Fetch data from SQL
    (entities, relations, attributes, keys, cursor_bn,
     cursor_setup, cursor, connection, connection_setup,
     connection_bn, db_name, db) = fetch_data_from_sql(args)

    # Create indices
    indices = create_indices(entities, keys)

    # Create mask matrices
    matrices = create_mask_matrices(relations, entities, indices, keys, args.device, db_name, cursor_setup)

    # Process rules and values
    (rules, multiples, states, functors, variables, nodes, masks,
     base_indices, mask_indices, sort_indices, stack_indices, values,
     prunes) = process_rules(cursor_bn, cursor_setup, relations, entities, keys,
                             indices, attributes, matrices, args)

    # Close database connections
    cursor.close()
    connection.close()
    cursor_setup.close()
    connection_setup.close()
    cursor_bn.close()
    connection_bn.close()

    return (rules, multiples, states, functors, variables, nodes, masks,
            base_indices, mask_indices, sort_indices, stack_indices, values,
            keys, indices, matrices, entities, attributes, relations, prunes)

def fetch_data_from_sql(args):
    database_name = {
        "cora": "cora",
        "citeseer": "citeseer",
        "imdb-multi": "imdb",
        "acm-multi": "acm-multi"
    }
    db_name = database_name[args.dataset]
    db = db_name
    host_name = 'database-3.cxcqxpvbnnwo.us-east-2.rds.amazonaws.com'
    user_name = "admin"
    password_name = "newPassword"

    connection = connect(host=host_name, user=user_name, password=password_name, db=db_name)
    cursor = connection.cursor()

    # Connect to setup database
    db_setup = db_name + "_setup"
    connection_setup = connect(host=host_name, user=user_name, password=password_name, db=db_setup)
    cursor_setup = connection_setup.cursor()

    # Connect to BN database
    db_bn = db_name + "_BN"
    connection_bn = connect(host=host_name, user=user_name, password=password_name, db=db_bn)
    cursor_bn = connection_bn.cursor()

    # Fetch entity tables
    cursor_setup.execute("SELECT TABLE_NAME FROM EntityTables")
    entity_tables = cursor_setup.fetchall()
    entities = {}
    keys = {}

    for (table_name,) in entity_tables:
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        cursor.execute(f"SHOW COLUMNS FROM {db}.{table_name}")
        columns = cursor.fetchall()
        columns_names = [column[0] for column in columns]
        entities[table_name] = DataFrame(rows, columns=columns_names)
        cursor_setup.execute("SELECT COLUMN_NAME FROM EntityTables WHERE TABLE_NAME = %s", (table_name,))
        key = cursor_setup.fetchall()
        keys[table_name] = key[0][0]

    # Fetch relation tables
    cursor_setup.execute("SELECT TABLE_NAME FROM RelationTables")
    relation_tables = cursor_setup.fetchall()
    relations = {}
    for (table_name,) in relation_tables:
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        cursor.execute(f"SHOW COLUMNS FROM {db}.{table_name}")
        columns = cursor.fetchall()
        columns_names = [column[0] for column in columns]
        relations[table_name] = DataFrame(rows, columns=columns_names)
        cursor_setup.execute("SELECT COLUMN_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = %s", (table_name,))
        key = cursor_setup.fetchall()
        keys[table_name] = (key[0][0], key[1][0])

    # Fetch attributes
    cursor_setup.execute("SELECT COLUMN_NAME, TABLE_NAME FROM AttributeColumns")
    attribute_columns = cursor_setup.fetchall()
    attributes = {}
    for column_name, table_name in attribute_columns:
        attributes[column_name] = table_name

    return (entities, relations, attributes, keys, cursor_bn, cursor_setup,
            cursor, connection, connection_setup, connection_bn, db_name, db)

def create_indices(entities, keys):
    indices = {}
    for table_name, df in entities.items():
        key = keys[table_name]
        indices[key] = {row[key]: idx for idx, row in df.iterrows()}
    return indices

def create_mask_matrices(relations, entities, indices, keys, device, db_name, cursor_setup):
    matrices = {}
    for table_name, df in relations.items():
        # Get foreign keys
        cursor_setup.execute("SELECT REFERENCED_TABLE_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = %s", (table_name,))
        reference = cursor_setup.fetchall()
        entity1 = reference[0][0]
        entity2 = reference[1][0]
        shape = (len(entities[entity1].index), len(entities[entity2].index))
        matrices[table_name] = torch.zeros(shape, dtype=torch.float32, device=device)

    for table_name, df in relations.items():
        cursor_setup.execute("SELECT COLUMN_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = %s", (table_name,))
        key = cursor_setup.fetchall()
        cursor_setup.execute("SELECT COLUMN_NAME, REFERENCED_COLUMN_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = %s", (table_name,))
        reference = cursor_setup.fetchall()
        rows_indices = []
        cols_indices = []
        for index, row in df.iterrows():
            row_index = indices[reference[0][1]][row[key[0][0]]]
            col_index = indices[reference[1][1]][row[key[1][0]]]
            rows_indices.append(row_index)
            cols_indices.append(col_index)
        rows_indices_tensor = torch.tensor(rows_indices, dtype=torch.long)
        cols_indices_tensor = torch.tensor(cols_indices, dtype=torch.long)
        matrices[table_name][rows_indices_tensor, cols_indices_tensor] = 1

    return matrices

def process_rules(cursor_bn, cursor_setup, relations, entities, keys, indices, attributes, matrices, args):
    # Fetch distinct children from Final_Path_BayesNets_view
    cursor_bn.execute("SELECT DISTINCT child FROM Final_Path_BayesNets_view")
    childs = cursor_bn.fetchall()
    rules = []
    multiples = []
    states = []
    functors = {}
    variables = {}
    nodes = {}
    masks = {}
    base_indices = []
    mask_indices = []
    sort_indices = []
    stack_indices = []
    values = []
    prunes = []

    # Get relation names
    relation_names = tuple(relations.keys())

    for i in range(len(childs)):
        # Build the rule
        rule = [childs[i][0]]
        cursor_bn.execute("SELECT parent FROM Final_Path_BayesNets_view WHERE child = %s", (childs[i][0],))
        parents = cursor_bn.fetchall()
        for (parent,) in parents:
            if parent != '':
                rule.append(parent)
        rules.append(rule)

        # Determine if rule is multiple
        multiples.append(1 if len(rule) > 1 else 0)

        # Check if rule contains relations
        relation_check = any(',' in atom for atom in rule)

        functor = {}
        variable = {}
        node = {}
        state = []
        mask = {}
        unmasked_variables = []

        for j in range(len(rule)):
            fun = rule[j].split('(')[0]
            functor[j] = fun
            if ',' not in rule[j]:
                var = rule[j].split('(')[1][:-1]
                variable[j] = var
                node[j] = var[:-1]
                if not relation_check:
                    unmasked_variables.append(var)
                    state.append(0)
                else:
                    mas = []
                    for k in rule:
                        func = k.split('(')[0]
                        if func not in relation_names:
                            func = attributes.get(func, func)
                        if ',' in k and var in k:
                            var1, var2 = k.split('(')[1][:-1].split(',')
                            mas.append([func, var1, var2])
                            unmasked_variables.append(k.split('(')[1][:-1])
                    mask[j] = mas
                    state.append(1)
            else:
                unmasked_variables.append(rule[j].split('(')[1][:-1])
                if fun in relation_names:
                    state.append(2)
                else:
                    state.append(3)
        functors[i] = functor
        variables[i] = variable
        nodes[i] = node
        states.append(state)
        masks[i] = mask

        # Base indices and mask indices
        masked_variables = [unmasked_variables[0]]
        base_indice = [0]
        mask_indice = []
        for j in range(1, len(unmasked_variables)):
            mask_check = False
            for k in range(len(masked_variables)):
                if unmasked_variables[j] == masked_variables[k]:
                    mask_indice.append([k, j])
                    mask_check = True
                    break
            if not mask_check:
                base_indice.append(j)
                masked_variables.append(unmasked_variables[j])

        # Sort indices
        sort_indice, sorted_variables = create_sort_indices(masked_variables, relation_check, relation_names, attributes)

        base_indices.append(base_indice)
        mask_indices.append(mask_indice)
        sort_indices.append(sort_indice)

        # Stack indices
        stack_indice = create_stack_indices(sorted_variables)
        stack_indices.append(stack_indice)

        # Fetch values
        cursor_bn.execute(f"SELECT * FROM `{childs[i][0]}_CP`")
        value = cursor_bn.fetchall()
        if args.rule_prune and not args.rule_weight:
            pruned_value = []
            for j in value:
                size = len(j)
                if multiples[i]:
                    if 2 * j[size - 4] * (log(j[size - 3]) - log(j[size - 1])) - log(j[size - 4]) > 0:
                        pruned_value.append(j)
                else:
                    if 2 * int(j[size - 3]) * (log(j[size - 5]) - log(j[size - 1])) - log(int(j[size - 3])) > 0:
                        pruned_value.append(j)
            values.append(pruned_value)
        elif args.rule_prune and args.rule_weight:
            pruned_value = []
            prune = []
            for j in value:
                size = len(j)
                if multiples[i]:
                    p = 2 * j[size - 4] * (log(j[size - 3]) - log(j[size - 1])) - log(j[size - 4])
                    if p > 0:
                        pruned_value.append(j)
                        prune.append(p)
                else:
                    p = 2 * int(j[size - 3]) * (log(j[size - 5]) - log(j[size - 1])) - log(int(j[size - 3]))
                    if p > 0:
                        pruned_value.append(j)
                        prune.append(p)
            prunes.append(prune)
            values.append(pruned_value)
        elif not args.rule_prune and args.rule_weight:
            raise Exception('Rule weighting requires rule pruning to be enabled.')
        else:
            values.append(value)

    # Adjust matrices if necessary
    adjust_matrices(matrices, rules, entities)

    return (rules, multiples, states, functors, variables, nodes, masks,
            base_indices, mask_indices, sort_indices, stack_indices, values, prunes)

def create_sort_indices(masked_variables, relation_check, relation_names, attributes):
    sort_indice = []
    sorted_variables = []
    if not relation_check:
        sort_indice.append([False, 0])
        sorted_variables.append(masked_variables[0])
    else:
        # Implement permutation logic here
        indices_permutations = list(permutations(range(len(masked_variables))))
        variables_permutations = list(permutations(masked_variables))
        found_chain = False
        for idx_perm, var_perm in zip(indices_permutations, variables_permutations):
            indices_chain = []
            variables_chain = []
            first = var_perm[0].split(',')[0]
            second = var_perm[0].split(',')[1]
            indices_chain.append([False, idx_perm[0]])
            variables_chain.append(var_perm[0])
            untransposed_check = True
            for k in range(1, len(var_perm)):
                next_first = var_perm[k].split(',')[0]
                next_second = var_perm[k].split(',')[1]
                if second == next_first:
                    second = next_second
                    indices_chain.append([False, idx_perm[k]])
                    variables_chain.append(var_perm[k])
                elif second == next_second:
                    second = next_first
                    indices_chain.append([True, idx_perm[k]])
                    variables_chain.append(next_second + ',' + next_first)
                else:
                    untransposed_check = False
                    break
            if untransposed_check:
                sort_indice = indices_chain
                sorted_variables = variables_chain
                found_chain = True
                break
        if not found_chain:
            # Try transposing the first element
            indices_chain = []
            variables_chain = []
            first = var_perm[0].split(',')[1]
            second = var_perm[0].split(',')[0]
            indices_chain.append([True, idx_perm[0]])
            variables_chain.append(first + ',' + second)
            untransposed_check = True
            for k in range(1, len(var_perm)):
                next_first = var_perm[k].split(',')[0]
                next_second = var_perm[k].split(',')[1]
                if second == next_first:
                    second = next_second
                    indices_chain.append([False, idx_perm[k]])
                    variables_chain.append(var_perm[k])
                elif second == next_second:
                    second = next_first
                    indices_chain.append([True, idx_perm[k]])
                    variables_chain.append(next_second + ',' + next_first)
                else:
                    untransposed_check = False
                    break
            if untransposed_check:
                sort_indice = indices_chain
                sorted_variables = variables_chain
                found_chain = True
    return sort_indice, sorted_variables

def create_stack_indices(sorted_variables):
    stack_indices = []
    for j in range(1, len(sorted_variables)):
        second = sorted_variables[j].split(',')[1]
        for k in range(j -1, -1, -1):
            previous_first = sorted_variables[k].split(',')[0]
            if previous_first == second:
                stack_indices.append([k, j])
    return stack_indices

def adjust_matrices(matrices, rules, entities):
    # Adjust matrices based on unique relation functors
    relation_functors = [item for sublist in rules for item in sublist if ',' in item]
    unique_relation_functors = list(set(relation_functors))
    for relation_functor in unique_relation_functors:
        entities_involved = relation_functor.replace(')', '').split('(')[1].split(',')
        entities_clean = [entity[:-1] for entity in entities_involved]
        correct_shape = (len(entities[entities_clean[0]]), len(entities[entities_clean[1]]))
        matrix_name = relation_functor.split('(')[0]
        if matrices[matrix_name].shape != correct_shape:
            matrices[matrix_name] = matrices[matrix_name].t()


def iteration_function(args, rules, multiples, states, functors, variables, nodes, masks, base_indices, mask_indices, sort_indices, stack_indices, values, keys, indices, matrices, entities, attributes, relations,  prunes, reconstructed_x_slice, reconstructed_labels, mode):
    
    functor_value_dict = dict()
    counter = 0
    counter_c1 = 0
    motif_list = []

    for table in range(len(rules)):
        indexx = -1
        for table_row in values[table]:
            indexx += 1
            unmasked_matrices, functor_value_dict, counter, counter_c1 = compute_unmasked_matrices(
                 args, table, table_row, rules, multiples, states, functors, variables, nodes, masks, mode,
                indices, keys, entities, reconstructed_x_slice, reconstructed_labels, matrices, attributes, relations,
                functor_value_dict, counter, counter_c1
            )

            masked_matrices = compute_masked_matrices(unmasked_matrices, base_indices[table], mask_indices[table])

            sorted_matrices = compute_sorted_matrices(masked_matrices, sort_indices[table])

            stacked_matrices = compute_stacked_matrices(sorted_matrices, stack_indices[table], args)

            result = compute_result(stacked_matrices)

            if args.rule_weight:
                motif_list.append(torch.sum(result) * prunes[table][indexx]) 
            else:
                motif_list.append(torch.sum(result))

            # Clean up
            del unmasked_matrices, masked_matrices, sorted_matrices, stacked_matrices, result

    return motif_list

def compute_unmasked_matrices(args, table, table_row, rules, multiples, states, functors, variables, nodes, masks, mode,
                              indices, keys, entities, reconstructed_x_slice, reconstructed_labels, matrices, attributes, relations,
                              functor_value_dict, counter, counter_c1):
    unmasked_matrices = []
    for column in range(len(rules[table])): 
        functor = functors[table][column]
        table_functor_value = table_row[column + multiples[table]]
        tuple_mask_info = ('0', '0', '0')
        variable = '0'
        functor_value_dict_key = (table_functor_value, functor, variable, tuple_mask_info)
        
        if mode == 'metric_ground_truth':
            if states[table][column] != 1:
                if functor_value_dict.get(functor_value_dict_key) is not None:
                    matrix = functor_value_dict[functor_value_dict_key]
                    unmasked_matrices.append(matrix)
                    counter += 1
                    continue
        state = states[table][column]

        if state == 0:
            matrix = compute_state_zero(
                 args, functor, table_functor_value, nodes[table][column], keys, entities,
                reconstructed_x_slice, reconstructed_labels,indices, mode
            )
            unmasked_matrices.append(matrix)
            if mode == 'metric_ground_truth':
                functor_value_dict[functor_value_dict_key] = matrix
        elif state == 1:
            matrices_list, functor_value_dict, counter, counter_c1 = compute_state_one(
                 args, functor, table_functor_value, variables[table][column], nodes[table][column], keys,
                entities, masks[table][column], indices, matrices, reconstructed_x_slice, reconstructed_labels, mode,
                functor_value_dict, counter, counter_c1
            )
            unmasked_matrices.extend(matrices_list)
        elif state == 2:
            matrix = compute_state_two(functor, table_functor_value, matrices)
            unmasked_matrices.append(matrix)
            functor_value_dict[functor_value_dict_key] = matrix
        elif state == 3:
            matrix = compute_state_three(
                functor, table_functor_value, attributes, keys, relations, indices, matrices
            )
            unmasked_matrices.append(matrix)
            functor_value_dict[functor_value_dict_key] = matrix
    return unmasked_matrices, functor_value_dict, counter, counter_c1

def compute_state_zero( args, functor, table_functor_value, functor_address, keys, entities, reconstructed_x_slice, reconstructed_labels, indices, mode):
    if mode == 'metric_ground_truth':
        primary_key = keys[functor_address]
        matrix = torch.zeros((len(entities[functor_address].index), 1), device=args.device)
        for entity_index in range(len(entities[functor_address][functor])):
            functor_value = entities[functor_address][functor][entity_index]
            if isinstance(table_functor_value, str):
                if isinstance(functor_value, (np.int64, np.int32)):
                    functor_value = str(functor_value)
                elif isinstance(functor_value, (np.float64, np.float32)):
                    functor_value = str(int(functor_value))    
            if functor_value == table_functor_value:
                key_index = entities[functor_address][primary_key][entity_index]
                row_index = indices[primary_key][key_index]
                matrix[row_index][0] = 1
    else:
        if '_' in functor:
            table_functor_value = int(table_functor_value)
            indx = int(functor[-1])-1
            if args.graph_type == 'heterogeneous':
                if table_functor_value == 0:
                    matrix = (1 - reconstructed_x_slice[functor_address[:-1]][:, indx]).float().view(-1, 1)
                else:
                    matrix = reconstructed_x_slice[functor_address[:-1]][:, indx].float().view(-1, 1)
            else:
                if table_functor_value == 0:
                    matrix = (1 - reconstructed_x_slice[:, indx]).float().view(-1, 1)
                else:
                    matrix = reconstructed_x_slice[:, indx].float().view(-1, 1)
        else:
            matrix = reconstructed_labels[:, int(table_functor_value)].float().view(-1, 1).to(args.device)
    return matrix

def compute_state_one(args, functor, table_functor_value, variable, functor_address, keys, entities, masks_list, indices, matrices,
                      reconstructed_x_slice, reconstructed_labels, mode, functor_value_dict, counter, counter_c1):
    matrices_list = []
    primary_key = keys[functor_address]
    for mask_info in masks_list:
        tuple_mask_info = tuple(mask_info)
        functor_value_dict_key = (table_functor_value, functor, variable, tuple_mask_info)
        if mode == 'metric_ground_truth':
            if functor_value_dict.get(functor_value_dict_key) is not None:
                matrix = functor_value_dict[functor_value_dict_key]
                matrices_list.append(matrix)
                counter += 1
                counter_c1 += 1
                continue
            if variable == mask_info[1]:
                matrix = torch.zeros((matrices[mask_info[0]].shape[0], 1), device=args.device)
            else:
                matrix = torch.zeros((1, matrices[mask_info[0]].shape[1]), device=args.device)
            for entity_index in range(len(entities[functor_address][functor])):
                functor_value = entities[functor_address][functor][entity_index]
                if functor_value == table_functor_value:
                    key_index = entities[functor_address][primary_key][entity_index]
                    index = indices[primary_key][key_index]
                    if variable == mask_info[1]:
                        matrix[index, 0] = 1
                    else:
                        matrix[0, index] = 1
            matrices_list.append(matrix)
            functor_value_dict[functor_value_dict_key] = matrix
        else:
            if variable == mask_info[1]:
                matrix = compute_state_one_variable(
                     args, functor, table_functor_value, functor_address, reconstructed_x_slice, reconstructed_labels
                )
            else:
                matrix = compute_state_one_variable_transpose(
                     args, functor, table_functor_value, functor_address, reconstructed_x_slice, reconstructed_labels
                )
            matrices_list.append(matrix)
    return matrices_list, functor_value_dict, counter, counter_c1

def compute_state_one_variable( args, functor, table_functor_value, functor_address, reconstructed_x_slice, reconstructed_labels):
    if '_' in functor:
        table_functor_value = int(table_functor_value)
        indx = int(functor[-1])-1
        if args.graph_type == 'heterogeneous':
            if table_functor_value == 0:
                matrix = (1 - reconstructed_x_slice[functor_address[:-1]][:, indx].float()).view(-1, 1)
            else:
                matrix = reconstructed_x_slice[functor_address[:-1]][:, indx].float().view(-1, 1)
        else:
            if table_functor_value == 0:
                matrix = (1 - reconstructed_x_slice[:, indx].float()).view(-1, 1)
            else:
                matrix = reconstructed_x_slice[:, indx].float().view(-1, 1)
    else:
        matrix = reconstructed_labels[:, int(table_functor_value)].float().view(-1, 1)
    return matrix

def compute_state_one_variable_transpose(args, functor, table_functor_value, functor_address, reconstructed_x_slice, reconstructed_labels):
    if '_' in functor:
        table_functor_value = int(table_functor_value)
        indx = int(functor[-1])-1
        if args.graph_type == 'heterogeneous':
            if table_functor_value == 0:
                matrix = (1 - reconstructed_x_slice[functor_address[:-1]][:, indx]).view(1, -1)
            else:
                matrix = reconstructed_x_slice[functor_address[:-1]][:, indx].view(1, -1)
        else:
            if table_functor_value == 0:
                matrix = (1 - reconstructed_x_slice[:, indx]).view(1, -1)
            else:
                matrix = reconstructed_x_slice[:, indx].view(1, -1)
    else:
        matrix = reconstructed_labels[:, int(table_functor_value)].view(1, -1).to(args.sdevice)
    return matrix

def compute_state_two(functor, table_functor_value, matrices):
    if table_functor_value == 'F':
        matrix = 1 - matrices[functor]
    else:
        matrix = matrices[functor]
    return matrix

def compute_state_three(args, functor, table_functor_value, attributes, keys, relations, indices, matrices):
    table_name = attributes[functor]
    primary_key = keys[table_name]
    if table_functor_value == 'N/A':
        matrix = 1 - matrices[table_name]
    else:
        matrix = torch.zeros_like(matrices[table_name], device=args.device)
        for index_relation in range(len(relations[table_name][functor])):
            functor_value = relations[table_name][functor][index_relation]
            if functor_value == table_functor_value:
                pk0_value = relations[table_name][primary_key[0]][index_relation]
                pk1_value = relations[table_name][primary_key[1]][index_relation]
                index1 = indices[primary_key[0]][pk0_value]
                index2 = indices[primary_key[1]][pk1_value]
                matrix[index1, index2] = 1
    return matrix

def compute_masked_matrices(unmasked_matrices, base_indices, mask_indices):
    masked_matrices = [unmasked_matrices[k] for k in base_indices]
    for k in mask_indices:
        masked_matrices[k[0]] = torch.mul(masked_matrices[k[0]], unmasked_matrices[k[1]])
    return masked_matrices

def compute_sorted_matrices(masked_matrices, sort_indices):
    sorted_matrices = []
    for k in sort_indices:
        if k[0]:
            sorted_matrices.append(masked_matrices[k[1]].T)
        else:
            sorted_matrices.append(masked_matrices[k[1]])
    return sorted_matrices

def compute_stacked_matrices(sorted_matrices, stack_indices, args):
    stacked_matrices = sorted_matrices.copy()
    pop_counter = 0
    for k in stack_indices:
        for _ in range(k[1] - k[0] - pop_counter):
            stacked_matrices[k[0]] = torch.mm(stacked_matrices[k[0]], stacked_matrices[k[0] + 1])
            stacked_matrices.pop(k[0] + 1)
            pop_counter += 1
        stacked_matrices[k[0]] = torch.mul(
            stacked_matrices[k[0]],
            torch.eye(len(stacked_matrices[k[0]]), device=args.device)
        )
    return stacked_matrices

def compute_result(stacked_matrices):
    result = stacked_matrices[0]
    for k in range(1, len(stacked_matrices)):
        result = torch.mm(result, stacked_matrices[k])
    return result








def process_reconstructed_data( args, mapping_details, reconstructed_adjacency, reconstructed_x, important_feat_ids, matrices,reconstructed_labels):

    if args.graph_type == 'heterogeneous':
        edge_encoding_to_node_types = {v: k for k, v in mapping_details['edge_type_encoding'].items()}
        filtered_reconstruct_adj = []
        
        for idx, adj_matrix in enumerate(reconstructed_adjacency):
            node_types = edge_encoding_to_node_types[idx + 1]  
            src_type, dst_type = node_types

            src_start, src_end = mapping_details['node_type_to_index_map'][src_type]
            dst_start, dst_end = mapping_details['node_type_to_index_map'][dst_type]

            filtered_matrix = adj_matrix[src_start:src_end, dst_start:dst_end]
            filtered_reconstruct_adj.append(filtered_matrix)
        
        filtered_reconstruct_adj_tensors = [matrix.to(args.device) for matrix in filtered_reconstruct_adj]
        
        for filtered_matrix in filtered_reconstruct_adj_tensors:
            filtered_shape = filtered_matrix.shape 
            
            for key, matrix in matrices.items():
                if matrix.shape == filtered_shape or matrix.shape == filtered_matrix.T.shape:
                    if filtered_matrix.size() == matrices[key].size():
                        matrices[key] = filtered_matrix.to(args.device)
                    elif filtered_matrix.T.size() == matrices[key].size():
                        matrices[key] = filtered_matrix.T.to(args.device)
                    break 
                
                
        reconstructed_x_splits = {}
        
        for node_type, (start_idx, end_idx) in mapping_details['node_type_to_index_map'].items():
            reconstructed_x_splits[f"{node_type}"] = reconstructed_x[start_idx:end_idx,:].to(args.device)
            
        node_type_counts = {}

        for node_types in mapping_details['edge_type_encoding'].keys():
            for node_type in node_types:
                if node_type in node_type_counts:
                    node_type_counts[node_type] += 1
                else:
                    node_type_counts[node_type] = 1

        repeated_node_types = [node_type for node_type, count in node_type_counts.items() if count > 1]
        st_idx, en_idx = mapping_details['node_type_to_index_map'][repeated_node_types[0]]
        reconstructed_labels_m = reconstructed_labels[st_idx:en_idx].to(args.device)
        
    else:
        reconstructed_x_splits = reconstructed_x.to(args.device)
        key = list(matrices.keys())[0]
        matrices[key] = reconstructed_adjacency[0].to(args.device)
        reconstructed_labels_m = reconstructed_labels.to(args.device)

    return reconstructed_x_splits, matrices, reconstructed_labels_m


def update_matrices(args, matrices, mapping_details, pre_self_loop_train_adj):
    
    for i in range(len(pre_self_loop_train_adj)):
        pre_self_loop_train_adj[i] = torch.tensor(pre_self_loop_train_adj[i])

    for key, matrix in matrices.items():
        entities = key.split('_')
        entity1, entity2 = entities[0][:-1], entities[1][:-1] 

        transpose = False
        if (entity1, entity2) in mapping_details['edge_type_encoding']:
            relation_number = mapping_details['edge_type_encoding'][(entity1, entity2)]
        elif (entity2, entity1) in mapping_details['edge_type_encoding']:
            relation_number = mapping_details['edge_type_encoding'][(entity2, entity1)]
            transpose = True  
        else:
            raise Exception(f"No relation found for key: {key}")

        relation_index = relation_number - 1

        if transpose:
            start_idx1, end_idx1 = mapping_details['node_type_to_index_map'][entity2]
            start_idx2, end_idx2 = mapping_details['node_type_to_index_map'][entity1]
        else:
            start_idx1, end_idx1 = mapping_details['node_type_to_index_map'][entity1]
            start_idx2, end_idx2 = mapping_details['node_type_to_index_map'][entity2]

        sliced_matrix = pre_self_loop_train_adj[relation_index][start_idx1:end_idx1, start_idx2:end_idx2]
        if transpose:
            sliced_matrix = sliced_matrix.T 
        

        matrices[key] = sliced_matrix.to(args.device)
        
        
def add_self_loops(matrices):
    for i in range(len(matrices)):
        np.fill_diagonal(matrices[i], 1)
    return matrices
        





