# utils/lr_scheduler.py
# Contains the poly learning rate calculation and the function to adjust optimizer groups

def lr_poly(base_lr, iter, max_iter, power=0.9):
    """Calculates the new learning rate based on polynomial decay."""
    return base_lr * ((1 - float(iter) / max_iter) ** power)

def adjust_learning_rate(optimizer, i_iter, max_iter, base_lr_rate):
    """Adjusts learning rate based on poly policy FOR ALL parameter groups."""
    lr = lr_poly(base_lr_rate, i_iter, max_iter)

    # Assuming first group is backbone (1x LR) and second is classifier (10x LR)
    optimizer.param_groups[0]['lr'] = lr 
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10 # Apply 10x LR to the second group

    # Return the base calculated LR for logging purposes if needed
    return lr
