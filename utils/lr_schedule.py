
def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=10, power=0.75, init_lr=0.001,weight_decay=0.0005, max_iter=10000):
    #10000
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    #max_iter = 10000
    gamma = 10.0
    lr = init_lr * (1 + gamma * min(1.0, iter_num / max_iter)) ** (-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i+=1
    return optimizer

