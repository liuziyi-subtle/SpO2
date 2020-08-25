# MIT License

# Copyright(c) 2020 Liu Ziyi

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import nni
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from scipy.io import loadmat


import logging

logger = logging.getLogger("auto-gbdt")

# specify your configurations as a dict


def get_default_parameters():
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 4,
        'metric': {'multi_error', 'multi_logloss'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    return params


def load_data():
    # 加载数据
    mat = loadmat("/data/data/spo2/raw/spo2_feature.mat")["f"]
    X = mat[:, :-1]
    y = mat[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y)

    split_num = int(len(X_train) * 0.8)
    X_eval = X_train[split_num:, :]
    X_train = X_train[:split_num, :]
    y_eval = y_train[split_num:]
    y_train = y_train[:split_num]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

    return lgb_train, lgb_eval, X_test, y_test


def run(lgb_train, lgb_eval, params, X_test, y_test):
    params["num_leaves"] = int(params["num_leaves"])

    gbm = lgb.train(params, lgb_train, num_boost_round=20,
                    valid_sets=lgb_eval, early_stopping_rounds=5)

    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred = [list(x).index(max(x)) for x in y_pred]

    # eval
    acc = accuracy_score(y_test, y_pred)
    print("The accuracy of prediction is: ", acc)

    nni.report_final_result(acc)


if __name__ == "__main__":
    # 加载数据lgb_train, lgb_eval为训练集和开发集, X_test, y_test为测试集
    lgb_train, lgb_eval, X_test, y_test = load_data()

    try:
        RECEIVED_PARAMS = nni.get_next_parameter()
        logger.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        logger.debug(PARAMS)

        # tain
        run(lgb_train, lgb_eval, PARAMS, X_test, y_test)

    except Exception as exception:
        logger.exception(exception)
        raise
