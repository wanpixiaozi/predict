# @title Default title text
import pandas as pd
import time
import matplotlib.pyplot as plt
import datetime
import numpy as np
import gc

# import the relevant Keras modules
# !pip install -q keras # this is not required if you are not using Google's colab
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout


def add_volatility(data, coins=['BTC', 'ETH']):
    """
    data: input data, pandas DataFrame
    coins: default is for 'btc and 'eth'. It could be changed as needed
    This function calculates the volatility and close_off_high of each given coin in 24 hours,
    and adds the result as new columns to the DataFrame.
    Return: DataFrame with added columns
    data：输入数据，pandas DataFrame
    coins：默认为'btc和'eth'。它可以根据需要进行更改
    此函数计算24小时内每枚给定代币的波动率和close/off/high，
    并将结果作为新列添加到DataFrame。
    返回：添加了列的DataFrame
    """
    for coin in coins:
        # calculate the daily change
        kwargs = {coin + '_change': lambda x: (x[coin + '_Close'] - x[coin + '_Open']) / x[coin + '_Open'],
                  coin + '_close_off_high': lambda x: 2 * (x[coin + '_High'] - x[coin + '_Close']) / (
                              x[coin + '_High'] - x[coin + '_Low']) - 1,
                  coin + '_volatility': lambda x: (x[coin + '_High'] - x[coin + '_Low']) / (x[coin + '_Open'])}
        data = data.assign(**kwargs)
    return data


def get_market_data(market, tag=True):
    """
    market：coinmarketcap.com上拼写的加密货币的全名。例如：'比特币'
    tag：例如：'btc'，如果提供的话，它会在每个列的名称中添加一个标签。
    返回：panda DataFrame
    此功能将使用coinmarketcap.com网址提供代币页面。
    阅读open/high/low/close/volume和市值。
    将日期格式转换为可读。
    通过将非数字值转换为非常接近0的数字，确保数据是一致的。
    最后标记每个列（如果提供）。
    """
    # market_data = pd.read_html("https://coinmarketcap.com/currencies/" + market +
    #                             "/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"), flavor='html5lib')[0]
    market_data = pd.read_html("https://coinmarketcap.com/currencies/" + market +
                               "/historical-data/?start=20180412&end="+time.strftime("%Y%m%d"), flavor='html5lib')[0]
    market_data.rename(columns={'Open*': 'Open', 'Close**': 'Close'}, inplace=True)
    market_data = market_data.assign(Date=pd.to_datetime(market_data['Date']))
    # print('transferred date market_data is', market_data)
    market_data['Volume'] = (pd.to_numeric(market_data['Volume'], errors='coerce').fillna(0))
    # print('transferred volume market_data is', market_data)
    if tag:
        market_data.columns = [market_data.columns[0]] + [tag + '_' + i for i in market_data.columns[1:]]
    print('mark tag transferred volume market_data is \n', market_data)
    return market_data


def create_model_data(data):
    """
    data: pandas DataFrame
    此函数会删除不必要的列，并根据降序日期反转DataFrame的顺序。
    Return: pandas DataFrame
    """
    # data = data[['Date']+[coin+metric for coin in ['btc_', 'eth_'] for metric in ['Close','Volume','close_off_high','volatility']]]
    data = data[['Date']+[coin+metric for coin in ['BTC_', 'ETH_'] for metric in ['Close','Volume']]]
    data = data.sort_values(by='Date')
    return data


def split_data(data, training_size=0.8):
    """
    data: Pandas Dataframe
    training_size: 用于训练的数据比例
    此函数根据给定的training_size将数据拆分为training_set和test_set
    返回：train_set和test_set为pandas DataFrame
    """
    return data[:int(training_size*len(data))], data[int(training_size*len(data)):]


def to_array(data):
    """
    data：DataFrame
    此函数将输入列表转换为numpy数组
    返回：numpy数组
    """
    x = [np.array(data[i]) for i in range (len(data))]
    return np.array(x)


def show_plot(data, tag):
    fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]})
    ax1.set_ylabel('Closing Price ($)',fontsize=12)
    ax2.set_ylabel('Volume ($ bn)',fontsize=12)
    ax2.set_yticks([int('%d000000000'%i) for i in range(10)])
    ax2.set_yticklabels(range(10))
    ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
    ax1.set_xticklabels('')
    ax2.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
    ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,7]])
    ax1.plot(data['Date'].astype(datetime.datetime),data[tag +'_Open'])
    ax2.bar(data['Date'].astype(datetime.datetime).values, data[tag +'_Volume'].values)
    fig.tight_layout()
    jpgfile = tag+ "_market.jpg"
    plt.savefig(jpgfile)
    plt.show()


class LSTM_Model:
    def __init__(self):
        self.neurons = 1024                 # number of hidden units in the LSTM layer
        self.activation_function = 'tanh'  # activation function for LSTM and Dense layer
        self.loss = 'mse'                  # loss function for calculating the gradient, in this case Mean Squared Error
        self.optimizer= 'adam'             # optimizer for applying gradient decent
        self.dropout = 0.25                # dropout ratio used after each LSTM layer to avoid over-fitting
        self.batch_size = 128
        self.epochs = 53

        # is an integer to be used as the look back window for creating a single input sample.
        self.window_len = 3
        self.training_size = 0.8           # proportion of data to be used for training

        # the earliest date which we have data for both ETH and BTC or any other provided coin
        self.merge_date = '2016-01-01'
        print('window_len is ', self.window_len)
        print('neurons is ', self.neurons)

    @staticmethod
    def merge_data(a, b, merge_date):
        """
        a: first DataFrame
        b: second DataFrame
        merge_date: includes the data from the provided date and drops the any data before that date.
        returns merged data as Pandas DataFrame
        merge_date：包含来自提供日期的数据，并删除该日期之前的任何数据。
        将合并数据作为Pandas DataFrame返回
        """
        merged_data = pd.merge(a, b, on=['Date'])
        merged_data = merged_data[merged_data['Date'] >= merge_date]
        return merged_data

    def create_inputs(self, data):
        """
        data: pandas DataFrame, 可以是training_set或test_set
        coins：将用作输入的代币数据。默认为'btc'，'eth'
        window_len：是一个用作创建单个输入样本的回溯窗口的整数。
        此函数将从给定数据集创建输入数组X，并将close和volume标准化为0到1之间
        返回：X，我们的模型作为python列表的输入，后来需要转换为numpy数组。
        """
        coins = ['BTC', 'ETH']
        norm_cols = [coin + metric for coin in coins for metric in ['_Close', '_Volume']]
        inputs = []
        for i in range(len(data) - self.window_len):
            temp_set = data[i:(i + self.window_len)].copy()
            inputs.append(temp_set)
            for col in norm_cols:
                inputs[i].loc[:, col] = inputs[i].loc[:, col] / inputs[i].loc[:, col].iloc[0] - 1
        return inputs

    def create_outputs(self, data, coin):
        """
        data：pandas DataFrame，可以是training_set或test_set
        coin：我们需要为其创建输出标签的目标硬币
        window_len：是一个用作创建单个输入样本的回溯窗口的integer。
        此函数将为我们的训练和验证创建标签数组，并在0和1之间对其进行标准化
        返回：标准化的numpy数组，用于给定硬币的“Close”价格
        """
        return (data[coin + '_Close'][self.window_len:].values / data[coin + '_Close'][:-self.window_len].values) - 1

    def build_model(self, inputs, output_size, neurons):
        """
        inputs: 输入数据为numpy数组
        output_size: 每个输入样本的预测数
        neurons: LSTM层中的神经元/单位数
        active_func: 要在LSTM图层和密集图层中使用的激活函数
        dropout: dropout ration, default is 0.25
        loss: 用于计算梯度的损失函数
        optimizer: 用于反向传播渐变的优化器类型
        This function will build 3 layered RNN model with LSTM cells with dropouts after each LSTM layer
        and finally a dense layer to produce the output using keras' sequential model.
        Return: Keras sequential model and model summary
        """
        dropout = self.dropout
        loss = self.loss
        optimizer = self.optimizer
        active_func = self.activation_function
        model = Sequential()
        model.add(LSTM(neurons, return_sequences=True, input_shape=(inputs.shape[1], inputs.shape[2]),
                       activation=active_func))

        model.add(Dropout(dropout))
        model.add(LSTM(neurons, return_sequences=True, activation=active_func))
        model.add(Dropout(dropout))
        model.add(LSTM(neurons, activation=active_func))
        model.add(Dropout(dropout))
        model.add(Dense(units=output_size))
        model.add(Activation(active_func))
        model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
        model.summary()
        return model

    def date_labels(self):
        last_date = self.market_data.iloc[0, 0]
        date_list = [last_date - datetime.timedelta(days=x) for x in range(len(self.X_test))]
        return[date.strftime('%m/%d/%Y') for date in date_list][::-1]

    def plot_results(self, history, model, y_target, coin):
        plt.figure(figsize=(25, 20))
        plt.subplot(311)
        plt.plot(history.epoch, history.history['loss'], )
        plt.plot(history.epoch, history.history['val_loss'])
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss')
        plt.title(coin + ' Model Loss')
        plt.legend(['Training', 'Test'])

        plt.subplot(312)
        plt.plot(y_target)
        plt.plot(model.predict(self.X_train))
        plt.xlabel('Dates')
        plt.ylabel('Price')
        plt.title(coin + ' Single Point Price Prediction on Training Set')
        plt.legend(['Actual','Predicted'])

        ax1 = plt.subplot(313)
        plt.plot(self.test_set[coin + '_Close'][self.window_len:].values.tolist())
        plt.plot(((np.transpose(model.predict(self.X_test)) + 1) * self.test_set[coin + '_Close'].values[:-self.window_len])[0])
        plt.xlabel('Dates')
        plt.ylabel('Price')
        plt.title(coin + ' Single Point Price Prediction on Test Set')
        plt.legend(['Actual','Predicted'])

        date_list = self.date_labels()
        ax1.set_xticks([x for x in range(len(date_list))])
        for label in ax1.set_xticklabels([date for date in date_list], rotation='vertical')[::2]:
            label.set_visible(False)

        jpgfile = coin + "_result.jpg"
        plt.savefig(jpgfile)
        plt.show()

    def main_process(self):
        run_btc = 1  # 默认只运行btc程序
        run_eth = 1
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        if run_btc:
            print('get btc data')
            btc_data = get_market_data("bitcoin", tag='BTC')

        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        if run_eth:
            print('get eth data')
            eth_data = get_market_data("ethereum", tag='ETH')

        if run_btc:
            btc_data.head()

        print('show plot')
        if run_btc:
            show_plot(btc_data, tag='BTC')
        if run_eth:
            show_plot(eth_data, tag='ETH')

        if run_eth:
            self.market_data = self.merge_data(btc_data, eth_data, self.merge_date)
        else:
            self.market_data = btc_data
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print('create_model_data')
        model_data = create_model_data(self.market_data)
        self.train_set, self.test_set = split_data(model_data)

        model_data.head()

        self.train_set = self.train_set.drop('Date', 1)
        self.test_set = self.test_set.drop('Date', 1)

        self.X_train = self.create_inputs(self.train_set)
        Y_train_btc = self.create_outputs(self.train_set, coin='BTC')
        self.X_test = self.create_inputs(self.test_set)
        Y_test_btc = self.create_outputs(self.test_set, coin='BTC')

        if run_eth:
            Y_train_eth = self.create_outputs(self.train_set, coin='ETH')
            Y_test_eth = self.create_outputs(self.test_set, coin='ETH')

        self.X_train, self.X_test = to_array(self.X_train), to_array(self.X_test)

        print(np.shape(self.X_train), np.shape(self.X_test), np.shape(Y_train_btc), np.shape(Y_test_btc))
        if run_eth:
            print(np.shape(self.X_train), np.shape(self.X_test), np.shape(Y_train_eth), np.shape(Y_test_eth))

        # clean up the memory
        gc.collect()

        # random seed for reproducibility
        np.random.seed(202)

        # initialise model architecture
        print('build model for btc')
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        btc_model = self.build_model(self.X_train, output_size=1, neurons=self.neurons)

        # train model on data
        print('train model on btc')
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        btc_history = btc_model.fit(self.X_train, Y_train_btc, epochs=self.epochs, batch_size=self.batch_size,
                                    verbose=1, validation_data=(self.X_test, Y_test_btc), shuffle=False)

        self.plot_results(btc_history, btc_model, Y_train_btc, coin='BTC')

        if run_eth:
            # clean up the memory
            gc.collect()

            # random seed for reproducibility
            np.random.seed(202)

            # initialise model architecture
            print('build model for eth')
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            eth_model = self.build_model(self.X_train, output_size=1, neurons=self.neurons)

            print('train model on eth')
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            # train model on data
            eth_history = eth_model.fit(self.X_train, Y_train_eth, epochs=self.epochs, batch_size=self.batch_size,
                                        verbose=1, validation_data=(self.X_test, Y_test_eth), shuffle=False)

            self.plot_results(eth_history, eth_model, Y_train_eth, coin='ETH')


if __name__ == '__main__':
    import sys, time

    temp = sys.stdout
    f = open('neoron1024-windowlen3.log', 'a')
    sys.stdout = f
    sys.stderr = f
    x = LSTM_Model()
    x.main_process()

    print('----------'*10)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('finished')
    f.close()
    sys.stdout = temp
    print('----------'*10)
    print('finished')
