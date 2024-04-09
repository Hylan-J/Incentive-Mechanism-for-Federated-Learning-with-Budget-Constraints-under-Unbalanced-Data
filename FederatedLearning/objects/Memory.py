class Memory:
    def __init__(self, dataset, aggregation_algorithm, incentive_mechanism, R, EMDs):
        # 实验设置数据
        self.experiment = f"{dataset}_{aggregation_algorithm}_{incentive_mechanism}_{R}"

        # 客户端质量数据
        self.client_EMDs = EMDs

        # 每轮交易及训练数据
        self.server_left_budget = []
        self.client_quotes = []
        self.client_accumulative_profits = []
        self.X = []
        self.P = []
        self.server_accuracy = []
        self.client_accuracies = []

    def add(self,
            server_left_budget: float,
            client_quotes: list,
            client_accumulative_profits: list,
            X: list,
            P: list,
            server_accuracy: float,
            client_accuracies: list):
        self.server_left_budget.append(server_left_budget)
        self.client_quotes.append(client_quotes)
        self.client_accumulative_profits.append(client_accumulative_profits)
        self.X.append(X)
        self.P.append(P)
        self.server_accuracy.append(server_accuracy)
        self.client_accuracies.append(client_accuracies)

    def save_excel(self):
        round_num = len(self.X)
        client_num = len(self.X[0])
        server_data = []
        client_data = []
        for i in range(round_num):
            server_data.append({"轮数": i + 1,
                                "剩余预算": self.server_left_budget[i],
                                "准确率": self.server_accuracy[i]})
            for j in range(client_num):
                client_data.append({"轮数": i + 1,
                                    "id": j + 1,
                                    "EMD": self.client_EMDs[j],
                                    "报价": self.client_quotes[i][j],
                                    "是否选中": "yes" if self.X[i][j] == 1 else "no",
                                    "成交价": self.P[i][j],
                                    "模型准确率": self.client_accuracies[i][j]})
        import xlwt
        workbook = xlwt.Workbook(encoding="utf-8")
        sheet1 = workbook.add_sheet("服务器")
        for col, column in enumerate(["轮数", "剩余预算", "准确率"]):
            sheet1.write(0, col, column)
        for row, data in enumerate(server_data):
            for col, col_data in enumerate(data):
                sheet1.write(row + 1, col, data[col_data])

        sheet2 = workbook.add_sheet("客户端")
        for col, column in enumerate(["轮数", "id", "EMD", "报价", "是否选中", "成交价", "模型准确率"]):
            sheet2.write(0, col, column)
        for row, data in enumerate(client_data):
            for col, col_data in enumerate(data):
                sheet2.write(row + 1, col, data[col_data])
        workbook.save("results/{}.xls".format(self.experiment))
