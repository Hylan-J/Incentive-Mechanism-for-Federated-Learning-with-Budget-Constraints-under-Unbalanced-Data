import random

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torchvision.datasets
from torchvision import transforms

from DeepLearning import *
from FederatedLearning import *

from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Num_Clients = 20
Num_Classes = 10
Num_Data = 1000

Dataset_Parameter = {
    "MNIST":
        {"batch_size": 10,
         "local_epochs": 5,
         "learning_rate": 0.01},
    "CIFAR10":
        {"batch_size": 10,
         "local_epochs": 5,
         "learning_rate": 0.1}
}


def init_prepare(dataset_type, divide_type):
    trainset, testset, model = None, None, None
    if dataset_type == "MNIST":
        trainset = torchvision.datasets.MNIST(root=DATASETS_ROOT_PATH,
                                              train=True,
                                              transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root=DATASETS_ROOT_PATH,
                                             train=False,
                                             transform=transforms.ToTensor())
        model = MNIST()
    elif dataset_type == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root='./DeepLearning/datasets',
                                                train=True,
                                                transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR10(root='./DeepLearning/datasets',
                                               train=False,
                                               transform=transforms.ToTensor())
        model = CIFAR10()

    hyperparameter = {
        "learning_rate": Dataset_Parameter[dataset_type]["learning_rate"],
        "local_epochs": Dataset_Parameter[dataset_type]["local_epochs"],
        "batch_size": Dataset_Parameter[dataset_type]["batch_size"]
    }

    EMDs = np.load(f"configs/EMDs_{divide_type}.npy")
    return trainset, testset, model, hyperparameter, EMDs.tolist()


def init_server(testset, model, hyperparameter):
    server = Server(testset=testset,
                    device=device,
                    global_net=deepcopy(model),
                    hyperparameter=hyperparameter)
    return server


def init_clients(EMDs, trainset, model, hyperparameter):
    distributions = [[]] * Num_Clients

    for i in range(Num_Clients):
        # distributed: flag, explain if data is allocated by rule
        # distribution: a list contains 10 elements, each element's index means class and value means number
        distributed, distribution = False, []
        # if is not allocated
        while not distributed:
            distributed, distribution = generate_distribution_by_EMD(EMD=EMDs[i],
                                                                     data_num=Num_Data,
                                                                     classes_num=Num_Classes)

        # get the allocation of data
        distribution = [int(distribution_) for distribution_ in distribution]
        distributions[i] = distribution
        EMDs[i] = cal_EMD(distribution / np.sum(distribution) * 1.0)

    print("\r-----------------------------------------------------")
    print("|\033[93m{:^51s}\033[0m|".format("local_nets init"))
    print("-----------------------------------------------------")
    # 二维list，第i个元素代表着数据集中第i类的索引list
    indexes_all = [torch.where(torch.Tensor(trainset.targets) == i)[0] for i in range(Num_Classes)]
    each_class_used = [0] * Num_Classes

    clients = []
    for i in range(Num_Clients):
        trainset_indexes = []
        for j in range(Num_Classes):
            if distributions[i][j] > 0:
                range_left = each_class_used[j]
                range_right = each_class_used[j] + distributions[i][j]
                trainset_indexes.extend(indexes_all[j][range_left:range_right])
                each_class_used[j] += distributions[i][j]

        client = Client(id=i,
                        trainset=torch.utils.data.Subset(trainset, trainset_indexes),
                        device=device,
                        local_net=deepcopy(model),
                        hyperparameter=hyperparameter)
        clients.append(client)
        print("| client {:2d} |     EMD: {:8f}     δ: {:8f}     |".format(i, EMDs[i], 1 / (e ** EMDs[i])))
    print("-----------------------------------------------------\n")
    return clients


def main(dataset, divide, aggregation_algorithm, incentive_mechanism, R):
    print("\r---------------------------------------------")
    print("|\033[93m{:^43s}\033[0m|".format("experiment info"))
    print("---------------------------------------------")
    print("|{:>25}  |  {:<13d}|".format("budget", R))
    print("|{:>25}  |  {}         |".format("device", device))
    print("|{:>25}  |  {:<13s}|".format("dataset", dataset))
    print("|{:>25}  |  {:<13d}|".format("data num", Num_Data))
    print("|{:>25}  |  {:<13d}|".format("clients num", Num_Clients))
    print("|{:>25}  |  {:<13d}|".format("classes num", Num_Classes))
    print("|{:>25}  |  {:<13s}|".format("incentive mechanism", incentive_mechanism))
    print("|{:>25}  |  {:<13s}|".format("aggregation algorithm", aggregation_algorithm))
    print("---------------------------------------------\n")

    result_accuracy = 0

    trainset, testset, model, hyperparameter, EMDs = init_prepare(dataset, divide)
    server = init_server(testset, model, hyperparameter)
    clients = init_clients(EMDs, trainset, model, hyperparameter)
    memory = Memory(dataset=dataset,
                    aggregation_algorithm=aggregation_algorithm,
                    incentive_mechanism=incentive_mechanism,
                    R=R,
                    EMDs=EMDs)

    client_accumulative_profits = [0.0] * Num_Clients

    client_accuracies = [0.0] * Num_Clients

    epoch = 1
    federated_learning_done = False
    while not federated_learning_done:
        print("\r----------------------------------------------------------------------------------")
        print("{:^86}".format("Epoch {:3d}  Budget:\033[93m{:6f}\033[0m").format(epoch, R))
        print("----------------------------------------------------------------------------------")
        client_quotes = [float(random.uniform(4 * 1 / e ** EMDs[i], 6 * 1 / e ** EMDs[i])) for i in range(Num_Clients)]

        bid_information = {
            "R": R,
            "c": client_quotes,
            "EMDs": EMDs
        }
        # print(client_quotes)
        X, P = globals()[incentive_mechanism](bid_information)

        # 如果存在被挑选中的客户端
        if sum(X) != 0:
            R -= sum(P)
            selected_local_nets = []
            for i in range(Num_Clients):
                if X[i] == 1:
                    clients[i].train()
                    client_accuracies[i] = server.evaluate(clients[i].local_net)
                    client_accumulative_profits[i] += P[i]

                    selected_local_nets.append(clients[i].local_net)
                    print(
                        "|client {:2d} | selected: {:1s}    quote: {:8f}    paid: {:8f}    profit: {:8f}|".
                        format(clients[i].id, "Y", client_quotes[i], P[i], client_accumulative_profits[i]))
                else:
                    client_accuracies[i] = 0
                    client_accumulative_profits[i] += 0
                    print(
                        "|client {:2d} | selected: {:1s}    quote: {:8f}    paid: {:8f}    profit: {:8f}|".
                        format(clients[i].id, "N", client_quotes[i], P[i], client_accumulative_profits[i]))

            ####################################################################################################
            # 使用聚合算法进行全局模型更新
            # ------------------------------------------------------------------------------------------------ #
            global_parameter = globals()[aggregation_algorithm](server.global_net, selected_local_nets)
            server.global_net.load_state_dict(global_parameter)
            # ------------------------------------------------------------------------------------------------ #
            server_accuracy = server.evaluate(server.global_net)
            print("|server model accuracy: \033[93m{:57s}\033[0m|".format(str(server_accuracy * 100) + "%"))
            result_accuracy = server_accuracy
            memory.add(server_left_budget=R,
                       client_quotes=client_quotes,
                       client_accumulative_profits=client_accumulative_profits,
                       X=X,
                       P=P,
                       server_accuracy=server_accuracy,
                       client_accuracies=client_accuracies)

            ####################################################################################################
            # 将服务器中的全局最优模型下发
            # ------------------------------------------------------------------------------------------------ #
            for client in clients:
                client.local_net.load_state_dict(server.global_net.state_dict())
            # ------------------------------------------------------------------------------------------------ #
            epoch += 1

        # 如果客户端不再参与, 则联邦学习过程结束
        else:
            federated_learning_done = True

    # memory.save_excel()
    print("----------------------------------------------------------------------------------\n")

    return result_accuracy


if __name__ == '__main__':
    Options_Dataset = ["MNIST", "CIFAR10"]
    Options_Divide = ["a", "b", "c"]
    Options_Aggregation_Algorithm = ["FedAvg"]
    Options_Incentive_Mechanism = ["FMore", "FLIM", "EMD_Greedy", "EMD_FLIM"]
    Options_R = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # 设置字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    # 该语句解决图像中的“-”负号的乱码问题
    plt.rcParams["axes.unicode_minus"] = False

    plot_data = {
        'FMore': [0.0] * len(Options_R),
        'FLIM': [0.0] * len(Options_R),
        'EMD_Greedy': [0.0] * len(Options_R),
        'EMD_FLIM': [0.0] * len(Options_R)
    }

    for i in Options_Incentive_Mechanism:
        for j in range(len(Options_R)):
            plot_data[i][j] = main(dataset=Options_Dataset[0],
                                   divide=Options_Divide[2],
                                   aggregation_algorithm=Options_Aggregation_Algorithm[0],
                                   incentive_mechanism=i,
                                   R=Options_R[j])
    # main(dataset=Options_Dataset[0],
    #      divide=Options_Divide[2],
    #      aggregation_algorithm=Options_Aggregation_Algorithm[0],
    #      incentive_mechanism=Options_Incentive_Mechanism[3],
    #      R=Options_R[0])

    plt.figure()
    plt.plot(plot_data['FMore'], 'o-', color='b',  label='FMore(truthfulness)')
    plt.plot(plot_data['FLIM'], '*-', color='r', label='FLIM(truthfulness)')
    plt.plot(plot_data['EMD_Greedy'], 'p-', color='y',  label='EMD-Greedy(truthfulness)')
    plt.plot(plot_data['EMD_FLIM'], 's-', color='g', label='EMD-FLIM(truthfulness)')
    plt.savefig('results.png', dpi=600)
    plt.show()
