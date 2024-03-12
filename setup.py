from pymysql import connect
from pandas import DataFrame
from numpy import zeros, int64, int32, float64, float32, multiply, dot, identity, sum
from itertools import permutations
from math import log
import torch





def setup_function(db_name, rule_prune):
    db = db_name 
    host_name = 'database-1.cxcqxpvbnnwo.us-east-2.rds.amazonaws.com'
    user_name = "admin"
    password_name = "newPassword"
    connection = connect(host=host_name, user=user_name, password=password_name, db=db_name)
    cursor = connection.cursor()
    db_setup = db_name + "_setup"
    connection_setup = connect(host=host_name, user=user_name, password=password_name, db=db_setup)
    cursor_setup = connection_setup.cursor()
    db_bn = db_name + "_BN"
    connection_bn = connect(host=host_name, user=user_name, password=password_name, db=db_bn)
    cursor_bn = connection_bn.cursor()
    
    
    
    keys = {}
    cursor_setup.execute("SELECT TABLE_NAME FROM EntityTables");
    entity_tables = cursor_setup.fetchall()
    entities = {}
    for i in entity_tables:
        cursor.execute("SELECT * FROM " + i[0])
        rows = cursor.fetchall()
        cursor.execute("SHOW COLUMNS FROM " + db + "." + i[0])
        columns = cursor.fetchall()
        entities[i[0]] = DataFrame(rows, columns=[columns[j][0] for j in range(len(columns))])
        cursor_setup.execute("SELECT COLUMN_NAME FROM EntityTables WHERE TABLE_NAME = " + "'" + i[0] + "'")
        key = cursor_setup.fetchall()
        keys[i[0]] = key[0][0]
        
        
        
        
    cursor_setup.execute("SELECT TABLE_NAME FROM RelationTables")
    relation_tables = cursor_setup.fetchall()
    relations = {}
    for i in relation_tables:
        cursor.execute("SELECT * FROM " + i[0])
        rows = cursor.fetchall()
        cursor.execute("SHOW COLUMNS FROM " + db + "." + i[0])
        columns = cursor.fetchall()
        relations[i[0]] = DataFrame(rows, columns=[columns[j][0] for j in range(len(columns))])
        cursor_setup.execute("SELECT COLUMN_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = " + "'" + i[0] + "'")
        key = cursor_setup.fetchall()
        keys[i[0]] = key[0][0], key[1][0]
        
    
    relation_names = tuple(i[0] for i in relation_tables)
    indices = {}
    for i in entity_tables:
        cursor_setup.execute("SELECT COLUMN_NAME FROM EntityTables WHERE TABLE_NAME = '" + i[0] + "'")
        key = cursor_setup.fetchall()[0][0]
        indices[key] = {}
        for index, row in entities[i[0]].iterrows():
            indices[key][row[key]] = index
            
            
    
    matrices = {}
    for i in relation_tables:
        cursor_setup.execute("SELECT REFERENCED_TABLE_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = '" + i[0] + "'")
        reference = cursor_setup.fetchall()
        shape = (len(entities[reference[0][0]].index), len(entities[reference[1][0]].index))
        matrices[i[0]] = torch.zeros(shape, dtype=torch.float32, device = 'cuda')  
    
    
    for i in relation_tables:
        cursor_setup.execute("SELECT COLUMN_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = '" + i[0] + "'")
        key = cursor_setup.fetchall()
        cursor_setup.execute("SELECT COLUMN_NAME, REFERENCED_COLUMN_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = '" + i[0] + "'")
        reference = cursor_setup.fetchall()
        
        rows_indices = []
        cols_indices = []
        for index, row in relations[i[0]].iterrows():
            row_index = indices[reference[0][1]][row[key[0][0]]]
            col_index = indices[reference[1][1]][row[key[1][0]]]
            rows_indices.append(row_index)
            cols_indices.append(col_index)
    
        rows_indices_tensor = torch.tensor(rows_indices, dtype=torch.long)
        cols_indices_tensor = torch.tensor(cols_indices, dtype=torch.long)
    
        matrices[i[0]][rows_indices_tensor, cols_indices_tensor] = 1
        
    
    cursor_setup.execute("SELECT COLUMN_NAME, TABLE_NAME FROM AttributeColumns")
    attribute_columns = cursor_setup.fetchall()
    attributes = {}
    for i in attribute_columns:
        attributes[i[0]] = i[1]
        
        
    
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
    for i in range(len(childs)):
        rule = [childs[i][0]]
        cursor_bn.execute("SELECT parent FROM Final_Path_BayesNets_view WHERE child = " + "'" + childs[i][0] + "'")
        parents = cursor_bn.fetchall()
        for j in parents:
            if j[0] != '':
                rule += [j[0]]
        rules.append(rule)

        if len(rule) == 1:
            multiples.append(0)
        else:
            multiples.append(1)
        relation_check = 0
        for j in rule:
            if j.find(',') != -1:
                relation_check = 1
        functor = {}
        variable = {}
        node = {}
        state = []
        mask = {}
        unmasked_variables = []
        for j in range(len(rule)):
            fun = rule[j].split('(')[0]
            functor[j] = fun
            if rule[j].find(',') == -1:
                var = rule[j].split('(')[1][:-1]
                variable[j] = var
                node[j] = var[:-1]
                if relation_check == 0:
                    unmasked_variables.append(var)
                    state.append(0)
    
                else:
                    mas = []
                    for k in rule:
                        func = k.split('(')[0]
                        if func not in relation_names:
                                func = attributes[func]
                        if k.find(',') != -1 and k.find(var) != -1:
                            unmasked_variables.append(k.split('(')[1][:-1])
                            mas.append([func, k.split('(')[1].split(',')[0], k.split('(')[1].split(',')[1][:-1]]) 
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
        masked_variables = [unmasked_variables[0]]
        base_indice = [0]
        mask_indice = []
        for j in range(1, len(unmasked_variables)):
            mask_check = 0
            for k in range(len(masked_variables)):
                if unmasked_variables[j] == masked_variables[k]:
                    mask_indice.append([k, j])
                    mask_check = 1
            if mask_check == 0:
                base_indice.append(j)
                masked_variables.append(unmasked_variables[j])
        sort_indice = []
        sorted_variables = []
        if relation_check == 0:
            sort_indice.append([False, 0])
            sorted_variables.append(masked_variables[0])
        else:
            indices_permutations = list(permutations(range(len(masked_variables))))
            variables_permutations = list(permutations(masked_variables))
            for j in range(len(variables_permutations)):
                indices_chain = []
                variables_chain = []
                first = variables_permutations[j][0].split(',')[0]
                second = variables_permutations[j][0].split(',')[1]
                indices_chain.append([False, indices_permutations[j][0]])
                variables_chain.append(variables_permutations[j][0])
                untransposed_check = 1
                transposed_check = 1
                if len(variables_permutations[j]) > 1:
                    for k in range(1, len(variables_permutations[j])):
                        next_first = variables_permutations[j][k].split(',')[0]
                        next_second = variables_permutations[j][k].split(',')[1]
                        if second == next_first:
                            second = next_second
                            indices_chain.append([False, indices_permutations[j][k]])
                            variables_chain.append(next_first + ',' + next_second)
                        elif second == next_second:
                            second = next_first
                            indices_chain.append([True, indices_permutations[j][k]])
                            variables_chain.append(next_second + ',' + next_first)    
                        else:
                            untransposed_check = 0
                            break
                    if untransposed_check != 1:
                        indices_chain[0] = [True, indices_permutations[j][0]]
                        variables_chain[0] = second + ',' + first
                        temp = first
                        first = second
                        second = temp
                        for k in range(1, len(variables_permutations[j])):
                            next_first = variables_permutations[j][k].split(',')[0]
                            next_second = variables_permutations[j][k].split(',')[1]
                            if second == next_first:
                                second = next_second
                                indices_chain.append([False, indices_permutations[j][k]])
                                variables_chain.append(next_first + ',' + next_second)
                            elif second == next_second:
                                second = next_first
                                indices_chain.append([True, indices_permutations[j][k]])
                                variables_chain.append(next_second + ',' + next_first)    
                            else:
                                transposed_check = 0
                                break
                if untransposed_check == 1 or transposed_check == 1 or len(variables_permutations[j]) == 1:
                    sort_indice = indices_chain
                    sorted_variables = variables_chain
                    break    
        stack_indice = []
        for j in range(1, len(sorted_variables)):
            second = sorted_variables[j].split(',')[1]
            for k in range(j - 1, -1, -1):
                previous_first = sorted_variables[k].split(',')[0]
                if previous_first == second:
                    stack_indice.append([k, j])   
        base_indices.append(base_indice)
        mask_indices.append(mask_indice)
        sort_indices.append(sort_indice)
        stack_indices.append(stack_indice)
        cursor_bn.execute("SELECT * FROM `" + childs[i][0] + "_CP`")
        value = cursor_bn.fetchall()
        if rule_prune == True :
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
        else:
            values.append(value)

        
    relation_functors = []
    for sublist in rules:
        for item in sublist:
            if item.count(',') == 1:
                relation_functors.append(item)

    unique_relation_functors = list(set(relation_functors))
    for relation_functor in unique_relation_functors:
        entities_involved = relation_functor.replace(')', '').split('(')[1].split(',')
    
        entities_clean = [entity[:-1] for entity in entities_involved]
    

        correct_shape = (len(entities[entities_clean[0]]), len(entities[entities_clean[1]]))
    

        if matrices[relation_functor.split('(')[0]].shape != correct_shape:
            matrices[relation_functor.split('(')[0]] = matrices[relation_functor.split('(')[0]].t()
       
        
    return rules, multiples, states, functors, variables, nodes, masks, base_indices, mask_indices, sort_indices, stack_indices, values, keys, indices, matrices, entities,attributes,relations 



def iteration_function(dataset, heterogeneous_data, rules, multiples, states, functors, variables, nodes, masks, base_indices, mask_indices, sort_indices, stack_indices, values, keys, indices, matrices, entities,attributes,relations, reconstructed_x_slice, reconstructed_labels,mode):

    functor_value_dict = dict()
    counter = 0
    counter_c1 = 0
    motif_list = []

    for table in range(len(rules)):
        if mode == 'predicted':
            print(rules[table])

        for table_row in values[table]:

            unmasked_matrices = []
            for column in range(len(rules[table])): 

                functor = functors[table][column]
                table_functor_value = table_row[column + multiples[table]]
                tuple_mask_info = ('0', '0', '0')
                variable = '0'
                functor_value_dict_key = (table_functor_value, functor, variable, tuple_mask_info)

                if(states[table][column] !=  1):
                    if functor_value_dict.get(functor_value_dict_key) != None:
                        matrix = functor_value_dict[functor_value_dict_key]
                        unmasked_matrices.append(matrix)
                        counter += 1
                        continue
                state = states[table][column]

                if state == 0:
                    if mode == 'ground_truth':
                        
                        functor_address = nodes[table][column]
                        primary_key = keys[functor_address]
    
                        matrix = torch.zeros((len(entities[functor_address].index), 1), device='cuda')
                        for entity_index in range(len(entities[functor_address][functor])):
                            functor_value = entities[functor_address][functor][entity_index]
                            if type(table_functor_value) == str:
                                if type(functor_value) == int64 or type(functor_value) == int32:
                                    functor_value = str(functor_value)
                                elif type(functor_value) == float64 or type(functor_value) == float32:
                                    functor_value = str(int(functor_value))    
                                
                            
                            if functor_value == table_functor_value:
                                key_index = entities[functor_address][primary_key][entity_index]
                                row_index = indices[primary_key][key_index]
                                matrix[row_index][0] = 1
                        unmasked_matrices.append(matrix)

                        functor_value_dict[functor_value_dict_key] = matrix
                    
                    else: 
                        functor_address = nodes[table][column]
                        primary_key = keys[functor_address]
                        if '_' in functor:
                            table_functor_value = int(table_functor_value)
                            if dataset in heterogeneous_data:

                                indx = int(functor[-1])-1
                                
                                if table_functor_value == 0:
                                    matrix = (1 - reconstructed_x_slice[functor_address[:-1]][:,indx].float()).view(-1, 1)
                                elif table_functor_value == 1:
                                    matrix = reconstructed_x_slice[functor_address[:-1]][:,indx].float().view(-1, 1)
                            else:
                                indx = int(functor[-1])-1
                                if table_functor_value == 0:
                                    matrix = (1 - reconstructed_x_slice[:,indx].float()).view(-1, 1)
                                elif table_functor_value == 1:
                                    matrix = reconstructed_x_slice[:,indx].float().view(-1, 1)
                                
                        else:
                            matrix = reconstructed_labels[:,int(table_functor_value)].float().to('cuda:0')
                        unmasked_matrices.append(matrix)

                        functor_value_dict[functor_value_dict_key] = matrix                        

                elif state == 1:
                    if mode == 'ground_truth':
                        variable = variables[table][column]
                        functor_address = nodes[table][column]
                        primary_key = keys[functor_address]
    
                        for mask_info in masks[table][column]:
                            tuple_mask_info = tuple(mask_info)
                            functor_value_dict_key = (table_functor_value, functor, variable, tuple_mask_info)
                            if functor_value_dict.get(functor_value_dict_key) is not None:
                                matrix = functor_value_dict[functor_value_dict_key]
                                unmasked_matrices.append(matrix)
                                counter += 1
                                counter_c1 += 1
                                continue
    
                            if variable == mask_info[1]:
                                matrix = torch.zeros((matrices[mask_info[0]].shape[0], 1), device='cuda')
                            elif variable == mask_info[2]:
                                matrix = torch.zeros((1, matrices[mask_info[0]].shape[1]), device='cuda')
    
                            for entity_index in range(len(entities[functor_address][functor])):
                                functor_value = entities[functor_address][functor][entity_index]
                                if functor_value == table_functor_value:
                                    key_index = entities[functor_address][primary_key][entity_index]
                                    index = indices[primary_key][key_index]
                                    if variable == mask_info[1]:
                                        matrix[index, 0] = 1
                                    elif variable == mask_info[2]:
                                        #print(key_index)
                                        matrix[0, index] = 1
                            unmasked_matrices.append(matrix)
                            functor_value_dict[functor_value_dict_key] = matrix
                    else:
                        
                        variable = variables[table][column]
                        functor_address = nodes[table][column]
                        primary_key = keys[functor_address]
        
                        for mask_info in masks[table][column]:
        
                            if variable == mask_info[1]:
                                if '_' in functor:
                                    table_functor_value = int(table_functor_value)
                                    if dataset in heterogeneous_data:
                                        indx = int(functor[-1])-1 
                                        if table_functor_value == 0:
                                            matrix = (1 - reconstructed_x_slice[functor_address[:-1]][:,indx].float()).view(-1, 1)
                                        elif table_functor_value == 1:
                                            matrix = reconstructed_x_slice[functor_address[:-1]][:,indx].float().view(-1, 1)
                                    else:
                                        indx = int(functor[-1])-1
                                        if table_functor_value == 0:
                                            matrix = (1 - reconstructed_x_slice[:,indx].float()).view(-1, 1)
                                        elif table_functor_value == 1:
                                            matrix = reconstructed_x_slice[:,indx].float().view(-1, 1)
                                        
                                else:
                                    matrix = reconstructed_labels[:,int(table_functor_value)].float().view(-1, 1)
                                unmasked_matrices.append(matrix)
                                functor_value_dict[functor_value_dict_key] = matrix              
                                
                                
                                
                            elif variable == mask_info[2]:
                                if '_' in functor:
                                    table_functor_value = int(table_functor_value)
                                    if dataset in heterogeneous_data:
                                        indx = int(functor[-1])-1 
                                        if table_functor_value == 0:
                                            matrix = (1 - reconstructed_x_slice[functor_address[:-1]][:,indx].float()).view(1,-1)
                                        elif table_functor_value == 1:
                                            matrix = reconstructed_x_slice[functor_address[:-1]][:,indx].float().view(1,-1)
                                    else:
                                        indx = int(functor[-1])-1
                                        if table_functor_value == 0:
                                            matrix = (1 - reconstructed_x_slice[:,indx].float()).view(1,-1)
                                        elif table_functor_value == 1:
                                            matrix = reconstructed_x_slice[:,indx].float().view(1,-1)
                                        
                                else:
                                    matrix = (reconstructed_labels[:,int(table_functor_value)].float().view(1,-1)).to('cuda:0')
                                unmasked_matrices.append(matrix)
                                functor_value_dict[functor_value_dict_key] = matrix  

                elif state == 2:
                    matrix = 1 - matrices[functor].float() if table_functor_value == 'F' else matrices[functor].float()
                    unmasked_matrices.append(matrix)
                    functor_value_dict[functor_value_dict_key] = matrix

                elif state == 3:
                    table_name = attributes[functor]
                    primary_key = keys[table_name]

                    if table_functor_value == 'N/A':
                        matrix = 1 - matrices[table_name]
                        unmasked_matrices.append(matrix)
                        functor_value_dict[functor_value_dict_key] = matrix

                    else:
                        matrix = torch.zeros_like(matrices[table_name], device='cuda')
                        for index_relation in range(len(relations[table_name][functor])):
                            functor_value = relations[table_name][functor][index_relation]
                            if functor_value == table_functor_value:
                                pk0_value = relations[table_name][primary_key[0]][index_relation]
                                pk1_value = relations[table_name][primary_key[1]][index_relation]
                                index1 = indices[primary_key[0]][pk0_value]
                                index2 = indices[primary_key[1]][pk1_value]
                                matrix[index1, index2] = 1
                        unmasked_matrices.append(matrix)
                        functor_value_dict[functor_value_dict_key] = matrix


            masked_matrices = []
            for k in base_indices[table]:
                masked_matrices.append(unmasked_matrices[k])

            for k in mask_indices[table]:
                masked_matrices[k[0]] = torch.mul(masked_matrices[k[0]], unmasked_matrices[k[1]])

            sorted_matrices = []
            for k in sort_indices[table]:
                if k[0]:
                    sorted_matrices.append(masked_matrices[k[1]].T)
                else:
                    sorted_matrices.append(masked_matrices[k[1]])
            stacked_matrices = sorted_matrices.copy()   
            pop_counter = 0

            for k in stack_indices[table]:
                for l in range(k[1] - k[0] - pop_counter):
                    stacked_matrices[k[0]] = torch.mm(stacked_matrices[k[0]], stacked_matrices[k[0] + 1])
                    stacked_matrices.pop(k[0] + 1)
                    pop_counter += 1
                stacked_matrices[k[0]] = torch.mul(stacked_matrices[k[0]], torch.eye(len(stacked_matrices[k[0]]), device='cuda'))
            result = stacked_matrices[0]

            for k in range(1, len(stacked_matrices)):
                result = torch.mm(result, stacked_matrices[k])


            motif_list.append(torch.sum(result))
            if mode == 'predicted':
                print(torch.sum(result))
            del unmasked_matrices, masked_matrices, sorted_matrices, stacked_matrices, matrix
 

    return motif_list




def process_reconstructed_data(dataset, heterogeneous_data, mapping_details, reconstructed_adjacency, reconstructed_x, important_feat_ids, matrices,reconstructed_labels):

    if dataset in heterogeneous_data:
        edge_encoding_to_node_types = {v: k for k, v in mapping_details['edge_type_encoding'].items()}
        filtered_reconstruct_adj = []
        
        for idx, adj_matrix in enumerate(reconstructed_adjacency):
            node_types = edge_encoding_to_node_types[idx + 1]  
            src_type, dst_type = node_types

            src_start, src_end = mapping_details['node_type_to_index_map'][src_type]
            dst_start, dst_end = mapping_details['node_type_to_index_map'][dst_type]

            filtered_matrix = adj_matrix[src_start:src_end, dst_start:dst_end]
            filtered_reconstruct_adj.append(filtered_matrix)
        
        filtered_reconstruct_adj_tensors = [matrix.to('cuda:0') for matrix in filtered_reconstruct_adj]
        
        for filtered_matrix in filtered_reconstruct_adj_tensors:
            filtered_shape = filtered_matrix.shape 
            
            for key, matrix in matrices.items():
                if matrix.shape == filtered_shape or matrix.t().shape == filtered_shape:
                    matrices[key] = filtered_matrix
                    break
                
        reconstructed_x_splits = {}
        
        for node_type, (start_idx, end_idx) in mapping_details['node_type_to_index_map'].items():
            reconstructed_x_splits[f"{node_type}"] = reconstructed_x[start_idx:end_idx,:].to('cuda:0')
            
        node_type_counts = {}

        for node_types in mapping_details['edge_type_encoding'].keys():
            for node_type in node_types:
                if node_type in node_type_counts:
                    node_type_counts[node_type] += 1
                else:
                    node_type_counts[node_type] = 1

        repeated_node_types = [node_type for node_type, count in node_type_counts.items() if count > 1]
        st_idx, en_idx = mapping_details['node_type_to_index_map'][repeated_node_types[0]]
        reconstructed_labels_m = reconstructed_labels[st_idx:en_idx].to('cuda:0')
        
    else:
        reconstructed_x_splits = reconstructed_x.to('cuda:0')
        key = list(matrices.keys())[0]
        matrices[key] = reconstructed_adjacency[0].to('cuda:0')
        reconstructed_labels_m = reconstructed_labels.to('cuda:0')

    return reconstructed_x_splits, matrices, reconstructed_labels_m





