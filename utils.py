from collections import defaultdict
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm
import tensorflow as tf


def pad_seq(x, max_len):
    if len(x) >= max_len:
        return x[:max_len]
    else:
        return [0] * (max_len - len(x)) + x


def pad_seq_with_start_end(x, max_len, start_token, end_token):
    if len(x) >= max_len:
        return [start_token] + x[:max_len] + [end_token]
    else:
        return [0] * (max_len - len(x)) + [start_token] + x + [end_token]


def get_session_data(data_dir, args):
    """
    Creates train-validation-test data sets for session based interactions
    There is a notion of session (or basket) and every interaction belongs
    to a session. Thus, the target is to predict all the items in the next
    session, which can be of variable length.

    Encoder-Decoder structure of the model is anticipated and the training
    data is prepared in a seq2seq format where the decoder has start and
    end of sequence tokens (in addition to the product/item tokens).

    If the number of past sessions used in the input is K then there are
    K+1 tensors in the input and one target tensor - one for each of the
    past session and the decoder input, which starts with a beginning of
    the sequence token. The target is a shifted version of this last inp
    that ends with a end of the sequence token.
    """

    num_past_sessions = args.num_past_sessions
    df_tr = pd.read_csv(os.path.join(data_dir, "transaction_data.csv"))
    dfg = df_tr.groupby("household_key")
    baskets = {}
    num_baskets = []
    num_products = []
    item_dict = {}
    print("Creating all the sessions from interaction data ...")
    for hkey, df_h in tqdm(dfg):
        df_hb = df_h.groupby("BASKET_ID")
        baskets[hkey] = {"products": [], "days": []}
        num_baskets.append(len(df_h["BASKET_ID"].unique()))
        for bid, df_basket in df_hb:
            products = df_basket["PRODUCT_ID"].tolist()
            for p in products:
                if p in item_dict:
                    item_dict[p] += 1
                else:
                    item_dict[p] = 1
            bday = list(set(df_basket["DAY"].tolist()))[0]
            baskets[hkey]["products"].append(products)
            baskets[hkey]["days"].append(bday)
            num_products.append(len(df_basket["PRODUCT_ID"].unique()))
    print(
        f"Total {len(baskets)} households with average {np.mean(num_baskets):.0f} baskets ({np.mean(num_products):.0f} products per basket)"
    )
    print(f"Total {len(item_dict)} items")

    filtered_items = set([k for k in item_dict if item_dict[k] > 10])
    print(len(filtered_items))

    new_baskets = {}
    for hhold in tqdm(range(1, 2500)):
        sessions = baskets[hhold]["products"]
        modified_sessions = []
        for sess in sessions:
            new_sess = [item for item in sess if item in filtered_items]
            if len(new_sess) > 0:
                modified_sessions.append(new_sess)
        if len(modified_sessions) > 0:
            new_baskets[hhold] = {"products": [], "days": []}
            new_baskets[hhold]["products"] = modified_sessions
        else:
            print(hhold)

    count_examples = 0
    count_items = {k: [] for k in range(num_past_sessions + 1)}
    for hhold in tqdm(range(1, 2500)):
        sessions = new_baskets[hhold]["products"]
        for ii in range(num_past_sessions, len(sessions)):
            inp, tgt = sessions[:ii], sessions[ii]
            for jj in range(num_past_sessions):
                count_items[jj].append(len(inp[jj]))
            count_items[jj + 1].append(len(tgt))
            count_examples += 1
    print(f"Total {count_examples} examples with {num_past_sessions} past sessions")
    print(
        "Minimum number of items per baskets:",
        [np.min(sess) for k, sess in count_items.items()],
    )
    print(
        "Average number of items per baskets:",
        [np.mean(sess) for k, sess in count_items.items()],
    )
    print(
        "Maximum number of items per baskets:",
        [np.max(sess) for k, sess in count_items.items()],
    )

    # Create the dictionary
    item_ids = {}
    considered_items = list(filtered_items)
    for ii in range(1, len(filtered_items) + 1):
        item_ids[considered_items[ii - 1]] = ii

    item_ids["START"] = ii + 1
    item_ids["END"] = ii + 2

    train_X, train_y = [[] for _ in range(args.num_past_sessions + 1)], []
    val_X, val_y = [[] for _ in range(args.num_past_sessions + 1)], []
    test_X, test_y = [[] for _ in range(args.num_past_sessions + 1)], []
    count_examples = 0
    train_val_test = [0.8, 0.1, 0.1]
    count_train, count_val, count_test = 0, 0, 0
    max_seq_len = 100
    start_token, end_token = item_ids["START"], item_ids["END"]

    print("Creating train, validation and test examples ...")
    for hhold in tqdm(range(1, 2500)):
        sessions = new_baskets[hhold]["products"]
        n_examples = len(sessions) - num_past_sessions
        n_train, n_val = int(n_examples * train_val_test[0]), int(
            n_examples * train_val_test[1]
        )
        # n_test = n_examples - n_train - n_val
        count = 0
        for ii in range(num_past_sessions, len(sessions)):
            inp, tgt = sessions[ii - num_past_sessions : ii], sessions[ii]

            # map to item-ids (1 ... #items)
            temp_x = []
            for exmp in inp:
                x_ = pad_seq([item_ids[jj] for jj in exmp], max_seq_len)
                temp_x.append(x_)

            # not padding, will be done later
            temp_y = [start_token] + [item_ids[jj] for jj in tgt] + [end_token]
            # temp_y = pad_seq_with_start_end(
            #     [item_ids[jj] for jj in tgt],
            #     max_seq_len - 1,
            #     start_token,
            #     end_token,
            # )

            if count < n_train:
                for kk in range(num_past_sessions):
                    train_X[kk].append(temp_x[kk])
                train_X[kk + 1].append(temp_y[:-1])  # decoder input
                train_y.append(temp_y[1:])  # decoder target
                count_train += 1
            elif count >= n_train + n_val:
                for kk in range(num_past_sessions):
                    test_X[kk].append(temp_x[kk])
                test_X[kk + 1].append(temp_y[:-1])  # decoder input
                test_y.append(temp_y[1:])
                count_test += 1
            else:
                for kk in range(num_past_sessions):
                    val_X[kk].append(temp_x[kk])
                val_X[kk + 1].append(temp_y[:-1])  # decoder input
                val_y.append(temp_y[1:])
                count_val += 1
            count_examples += 1
            count += 1
    print(
        f"Total {count_train} train, {count_val} validation and {count_test} test examples"
    )
    return train_X, train_y, val_X, val_y, test_X, test_y, item_ids


def session_data_partition(data_dir, args):
    """
    Creates train-validation-test data sets for session based interactions
    However, it is assumed that only one past session is available and as
    a result the input is a single vector of the last session items and the
    target is the next session items.

    It is assumed that the a SASRec kind of model will be employed that
    directly maps the input sequence to the target sequence with both having
    the same length.
    """

    num_past_sessions = args.num_past_sessions
    df_tr = pd.read_csv(os.path.join(data_dir, "transaction_data.csv"))
    dfg = df_tr.groupby("household_key")
    baskets = {}
    num_baskets = []
    num_products = []
    item_dict = {}
    print("Creating all the sessions from interaction data ...")
    for hkey, df_h in tqdm(dfg):
        df_hb = df_h.groupby("BASKET_ID")
        baskets[hkey] = {"products": [], "days": []}
        num_baskets.append(len(df_h["BASKET_ID"].unique()))
        for bid, df_basket in df_hb:
            products = df_basket["PRODUCT_ID"].tolist()
            for p in products:
                if p in item_dict:
                    item_dict[p] += 1
                else:
                    item_dict[p] = 1
            bday = list(set(df_basket["DAY"].tolist()))[0]
            baskets[hkey]["products"].append(products)
            baskets[hkey]["days"].append(bday)
            num_products.append(len(df_basket["PRODUCT_ID"].unique()))
    print(
        f"Total {len(baskets)} households with average {np.mean(num_baskets):.0f} baskets ({np.mean(num_products):.0f} products per basket)"
    )
    print(f"Total {len(item_dict)} items")

    filtered_items = set([k for k in item_dict if item_dict[k] > 10])
    print(len(filtered_items))

    new_baskets = {}
    for hhold in tqdm(range(1, 2500)):
        sessions = baskets[hhold]["products"]
        modified_sessions = []
        for sess in sessions:
            new_sess = [item for item in sess if item in filtered_items]
            if len(new_sess) > 0:
                modified_sessions.append(new_sess)
        if len(modified_sessions) > 0:
            new_baskets[hhold] = {"products": [], "days": []}
            new_baskets[hhold]["products"] = modified_sessions
        else:
            print(hhold)

    count_examples = 0
    count_items = {k: [] for k in range(num_past_sessions + 1)}
    for hhold in tqdm(range(1, 2500)):
        sessions = new_baskets[hhold]["products"]
        for ii in range(num_past_sessions, len(sessions)):
            inp, tgt = sessions[:ii], sessions[ii]
            for jj in range(num_past_sessions):
                count_items[jj].append(len(inp[jj]))
            count_items[jj + 1].append(len(tgt))
            count_examples += 1
    print(f"Total {count_examples} examples with {num_past_sessions} past sessions")
    print(
        "Minimum number of items per baskets:",
        [np.min(sess) for k, sess in count_items.items()],
    )
    print(
        "Average number of items per baskets:",
        [np.mean(sess) for k, sess in count_items.items()],
    )
    print(
        "Maximum number of items per baskets:",
        [np.max(sess) for k, sess in count_items.items()],
    )

    # Create the dictionary
    item_ids = {}
    considered_items = list(filtered_items)
    for ii in range(1, len(filtered_items) + 1):
        item_ids[considered_items[ii - 1]] = ii

    train_X, train_y = [], []
    val_X, val_y = [], []
    test_X, test_y = [], []
    count_examples = 0
    train_val_test = [0.8, 0.1, 0.1]
    count_train, count_val, count_test = 0, 0, 0

    print("Creating train, validation and test examples ...")
    for hhold in tqdm(range(1, 2500)):
        sessions = new_baskets[hhold]["products"]
        n_examples = len(sessions) - 1
        n_train, n_val = int(n_examples * train_val_test[0]), int(
            n_examples * train_val_test[1]
        )
        # n_test = n_examples - n_train - n_val
        count = 0
        for ii in range(1, len(sessions)):
            inp, tgt = sessions[ii - 1], sessions[ii]

            # map to item-ids (1 ... #items)
            temp_x = [item_ids[jj] for jj in inp]  # padding is done later
            temp_y = [item_ids[jj] for jj in tgt]

            if count < n_train:
                train_X.append(temp_x)
                train_y.append(temp_y)  # decoder target
                count_train += 1
            elif count >= n_train + n_val:
                test_X.append(temp_x)  # decoder input
                test_y.append(temp_y)
                count_test += 1
            else:
                val_X.append(temp_x)
                val_y.append(temp_y)
                count_val += 1
            count_examples += 1
            count += 1
    print(
        f"Total {count_train} train, {count_val} validation and {count_test} test examples"
    )
    return train_X, train_y, val_X, val_y, test_X, test_y, item_ids


def create_dataset(data_dir, args):
    """
    Creates train-validation-test data sets from interaction data.
    There is no notion of session/baskets and number of train,
    validation and test examples is same as the number of users.

    We want to predict the next few items (pre-decided) without
    using any negative sampling.
    """
    inp_file = os.path.join(data_dir, args.dataset + ".txt")
    sample = pd.read_csv(inp_file, sep=args.colsep, nrows=5)
    ncol = sample.shape[1]
    if ncol == 1:
        raise ValueError("Not enough data to unpack!!")

    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    train_X, train_y = [], []
    valid_X, valid_y = [], []
    test_X = []
    with open(inp_file, "r") as fr:
        for line in tqdm(fr):
            if ncol == 2:
                u, i = line.rstrip().split(args.colsep)
            elif ncol == 3:
                u, i, timestamp = line.rstrip().split(args.colsep)
            elif ncol == 4:
                u, i, _, _ = line.rstrip().split(args.colsep)
            else:
                raise ValueError("Unknown number of columns")
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        test_X.append(User[user])  # entire sequence, all users
        if nfeedback == 1:
            continue
        elif nfeedback == 2:
            train_X.append(User[user][:-1])
            train_y.append(User[user][-1])
        else:
            train_X.append(User[user][:-2])
            train_y.append(User[user][-2])

            valid_X.append(User[user][:-1])
            valid_y.append(User[user][-1])

    train_X = tf.keras.preprocessing.sequence.pad_sequences(
        train_X, padding="pre", truncating="pre", maxlen=args.maxlen
    )
    valid_X = tf.keras.preprocessing.sequence.pad_sequences(
        valid_X, padding="pre", truncating="pre", maxlen=args.maxlen
    )
    test_X = tf.keras.preprocessing.sequence.pad_sequences(
        test_X, padding="pre", truncating="pre", maxlen=args.maxlen
    )

    train_y = np.array(train_y)
    valid_y = np.array(valid_y)

    return train_X, train_y, valid_X, valid_y, test_X, usernum, itemnum
