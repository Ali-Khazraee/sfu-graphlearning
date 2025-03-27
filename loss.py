import torch
import torch.nn as nn
import torch.nn.functional as F

# objective Function
def OptimizerVAE(args, important_feat_ids, pred, labels, std_z, mean_z, num_nodes, pos_wight, norm, x_pred, x_true, observed, predicted, predicted_node_labels, gt_labels, indexes_to_ignore=None, val_edge_idx=None):
    """
    :param pred: reconstructed adj matrix by model; its a stack of dense adj matrix
    :param labels: The origianl adj matrix which shoul be reconstructed; stack of matrices
    :param std_z:
    :param mean_z:
    :param num_nodes:
    :param pos_wight: The ratio of positive to negative edges
    :param norm:
    :param indexes_to_ignore: This edges whould not effect the reconstruction loss; its a list  of lists with len(indexes_to_ignore)==2; e.g   [indexes_to_ignore[0],indexes_to_ignore[1]] wount be effect the reconstruction loss
    :param val_edge_idx: the indexces of val edges. its just for cal reconstrction loss of val edges
    :return:
    """
    val_recons_loss = None

    # loss for motif counting 
    
    if args.motif_obj == True: 



        zero_indices = [i for i, t in enumerate(observed) if torch.any(t == 0)]

        filtered_observed = [g for i, g in enumerate(observed) if i not in zero_indices]
        filtered_predicted = [p for i, p in enumerate(predicted) if i not in zero_indices]

        normalized_observed = [torch.ones_like(t) for t in filtered_observed]

        # normalized_predicted = [p / g for p, g in zip(filtered_predicted, filtered_observed)]

        normalized_predicted = [torch.abs((torch.log(p / g))) for p, g in zip(filtered_predicted, filtered_observed)]
        
        motif_loss = (((torch.sum(torch.stack(normalized_predicted))/len((normalized_predicted)))))
    else: 
        motif_loss = 0
    
    
    reconstruction_loss = norm * F.binary_cross_entropy_with_logits(pred, labels, pos_weight=pos_wight, reduction='none')

    # the validation reconstruction loss
    if val_edge_idx:
        val_recons_loss = reconstruction_loss[:, val_edge_idx[0], val_edge_idx[1]].mean()

    # some edges wont have reconstruciton losss
    if indexes_to_ignore and len(indexes_to_ignore) > 0:
        reconstruction_loss[:, indexes_to_ignore[0], indexes_to_ignore[1]] = 0  # masking edges

    reconstruction_loss = reconstruction_loss.mean()


    # KL divergence
    kl_loss = (-0.5 / num_nodes) * torch.mean(
        torch.sum(1 + 2 * torch.log(std_z) - mean_z.pow(2) - (std_z).pow(2), dim=1))
    feat_loss = F.binary_cross_entropy_with_logits(x_pred,x_true[:,important_feat_ids].float())
    

    
    # label loss
    not_masked_labels = torch.where((gt_labels != -1 ))[0]
    criterion = nn.CrossEntropyLoss()
    label_loss = criterion(predicted_node_labels[not_masked_labels,:], gt_labels[not_masked_labels])


    acc = (torch.sigmoid(pred).round() == labels).sum() / float(pred.shape[0] * pred.shape[1]*pred.shape[2]) # accuracy on the train data

    loss = reconstruction_loss + kl_loss + label_loss + motif_loss
    if args.motif_obj :
        loss += motif_loss * args.motif_weight
    

    return kl_loss, reconstruction_loss, feat_loss , acc, val_recons_loss , motif_loss, label_loss , loss

    
