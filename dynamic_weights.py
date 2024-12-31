import torch


def dy_weight(loss1=0, loss2=0, w1=0.5, w2=0.5, alpha=10, beta=0.35):
    '''
    动态权重计算
    :param loss1: 模型一的损失
    :param loss2: 模型二的损失
    :param w1: 模型一在 t-1 时刻的权重
    :param w2: 模型二在 t-1 时刻的权重
    :param alpha: 调节参数
    :param beta: 调节参数,防止一个权重过大
    :return: 模型一和模型二在 t 时刻的权重
    '''
    temp1 = w1 + 1 / (1 + alpha * loss1)
    temp2 = w2 + 1 / (1 + alpha * loss2)
    if ((temp1 - temp2) / (temp1 + temp2) > beta
            or (temp2 - temp1) / (temp1 + temp2) > beta):
        # 防止权重差距过大
        return [w1, w2]
    else:
        w1 = temp1 / (temp1 + temp2)
        w2 = temp2 / (temp1 + temp2)
        return [w1, w2]


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        p = p.type(torch.long)
        t = t.type(torch.long)
        conf_matrix[p, t] += 1
    return conf_matrix
