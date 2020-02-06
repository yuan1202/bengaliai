import torch


def to_onehot(truth, num_class):
    batch_size = len(truth)
    onehot = torch.zeros(batch_size,num_class).to(truth.device)
    onehot.scatter_(dim=1, index=truth.view(-1,1),value=1)
    return onehot


def cross_entropy_onehot_loss(logit, onehot):
    batch_size,num_class = logit.shape
    log_probability = -F.log_softmax(logit,1)
    loss = (log_probability*onehot)
    loss = loss.sum(1)
    loss = loss.mean()
    return loss


def criterion(logit, truth):
    loss = []
    for l,t in zip(logit,truth):
        #e = F.cross_entropy(l, t)
        e = cross_entropy_onehot_loss(l, t)
        loss.append(e)

    return loss

#https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
def metric(probability, truth):

    correct = []
    for p,t in zip(probability,truth):
        p = p.data.cpu().numpy()
        t = t.data.cpu().numpy()
        y = p.argmax(-1)
        c = np.mean(y==t)
        correct.append(c)

    return correct


def logit_to_probability(logit):
    probability = [ F.softmax(l,1) for l in logit ]
    return probability