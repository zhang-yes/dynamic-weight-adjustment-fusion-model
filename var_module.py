from dynamic_weights import confusion_matrix

import torch


def data_normal_2d(orign_data, dim="col"):
    """
	针对于2维tensor归一化
	可指定维度进行归一化，默认为行归一化
	参数1为原始tensor，参数2为默认指定行，输入其他任意则为列
    """
    if dim == "col":
        dim = 1
        d_min = torch.min(orign_data, dim=dim)[0]
        for idx, j in enumerate(d_min):
            if j < 0:
                orign_data[idx, :] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data, dim=dim)[0]
    else:
        dim = 0
        d_min = torch.min(orign_data, dim=dim)[0]
        for idx, j in enumerate(d_min):
            if j < 0:
                orign_data[:, idx] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data, dim=dim)[0]
    d_max = torch.max(orign_data, dim=dim)[0]
    dst = d_max - d_min
    if d_min.shape[0] == orign_data.shape[0]:
        d_min = d_min.unsqueeze(1)
        dst = dst.unsqueeze(1)
    else:
        d_min = d_min.unsqueeze(0)
        dst = dst.unsqueeze(0)
    norm_data = torch.sub(orign_data, d_min).true_divide(dst)
    # norm_data = (norm_data - 0.5).true_divide(0.5)
    return norm_data


# x = torch.randint(low=-10,high=10,size=(3,6))
# norm_data = data_normal_2d(x)
# # tensor([[1.0000, 0.8182, 0.4545, 0.0000, 0.8182, 0.2727],
# #         [0.6154, 0.5385, 0.5385, 0.0769, 0.0000, 1.0000],
# #         [0.4444, 0.3889, 0.0000, 0.2778, 1.0000, 0.5000]])
# print(norm_data)


def var_module(model, loss_function, input, label, matrix_input):
    valid_loss = 0
    valid_acc = 0
    output = model(input)

    # softmax = torch.nn.Softmax(dim=1)
    output_normal = data_normal_2d(output)

    loss = loss_function(output, label)
    valid_loss += loss.item() * input.size(0)
    ret, pred = torch.max(output.data, 1)
    matrix_output = confusion_matrix(output, label, matrix_input)
    correct_counts = pred.eq(label.data.view_as(pred))
    acc = torch.mean(correct_counts.type(torch.FloatTensor))
    valid_acc += acc.item() * input.size(0)
    var_data_len = len(input)
    if var_data_len <= 0:
        var_data_len = 1
    return valid_loss, valid_acc, matrix_output, output, var_data_len, output_normal,pred
