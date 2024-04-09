from copy import deepcopy


def FedAvg(global_net, local_nets):
    # 将客户端模型参数取平均并更新全局模型
    global_parameter = deepcopy(global_net.state_dict())
    local_parameters = [deepcopy(local_net.state_dict()) for local_net in local_nets]

    for layer_name in global_parameter.keys():
        global_parameter[layer_name] = sum([local_parameter[layer_name] for local_parameter in local_parameters])
        global_parameter[layer_name] = global_parameter[layer_name] / len(local_parameters)
    return global_parameter
