import os
import time
import argparse
import pandas as pd
import math
import numpy as np
import pickle
from tqdm import tqdm
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

from models.transformer_tf import build_multilevel_transformer_unequal

from models.sasrec import SASREC
from models.sasrec_multi_session import SASREC as SASREC2
from models.tencoder import TENCODER
from utils import get_session_data, session_data_partition, create_dataset
from sampler import WarpSampler


def create_combined_dataset_sasrec(u, seq, pos, neg, extras):
    params = extras[0]
    seq_max_len = params.maxlen
    inputs = {}

    seq = tf.keras.preprocessing.sequence.pad_sequences(
        seq, padding="pre", truncating="pre", maxlen=seq_max_len
    )
    pos = tf.keras.preprocessing.sequence.pad_sequences(
        pos, padding="pre", truncating="pre", maxlen=seq_max_len
    )
    neg = tf.keras.preprocessing.sequence.pad_sequences(
        neg, padding="pre", truncating="pre", maxlen=seq_max_len
    )

    inputs["users"] = np.expand_dims(np.array(u), axis=-1)
    inputs["input_seq"] = seq
    inputs["positive"] = pos
    inputs["negative"] = neg

    target = np.concatenate(
        [
            np.repeat(1, seq.shape[0] * seq.shape[1]),
            np.repeat(0, seq.shape[0] * seq.shape[1]),
        ],
        axis=0,
    )
    target = np.expand_dims(target, axis=-1)
    return inputs, target


def create_combined_dataset(seqs, tgt, extras):
    params = extras[0]
    seq_max_len = params.maxlen
    inputs = {}

    if params.num_past_sessions == 1:
        seq = tf.keras.preprocessing.sequence.pad_sequences(
            seqs, padding="pre", truncating="pre", maxlen=seq_max_len
        )
        inputs["input_seq1"] = seq

    elif params.num_past_sessions > 1:
        for ii, seq in enumerate(seqs):
            seq = tf.keras.preprocessing.sequence.pad_sequences(
                seq, padding="pre", truncating="pre", maxlen=seq_max_len
            )
            inputs["input_seq" + str(ii + 1)] = seq

    tgt = tf.keras.preprocessing.sequence.pad_sequences(
        tgt, padding="pre", truncating="pre", maxlen=seq_max_len
    )
    inputs["target_seq"] = tgt

    return inputs


def get_jaccard_score(x1, x2, ignore=None):
    """Computes the Jaccard Score between two lists
    ignore can be a list of elements to be excluded
    while comparing

    """
    s1 = set(x1)
    s2 = set(x2)
    if ignore:
        s1 -= set(ignore)
        s2 -= set(ignore)

    num = len(s1.intersection(s2))
    den = len(s1.union(s2))
    return num / den


def get_batch_score(target, pred, ignore=[0]):
    all_scores = []
    for ii in range(target.shape[0]):
        s = get_jaccard_score(list(target[ii, :]), list(pred[ii, :]), ignore)
        all_scores.append(s)
    return np.mean(all_scores)


def evaluate_sasrec(model, sampler, num_examples, args):
    num_steps, rem = int(num_examples / args.batch_size), int(
        num_examples % args.batch_size
    )
    if rem > 0:
        num_steps += 1
    all_scores = []
    for step in tqdm(
        range(num_steps), total=num_steps, ncols=70, leave=False, unit="b"
    ):
        u, seq, pos, negs = sampler.next_batch()
        # each element is of length = batch_size
        # negs =
        # print(len(negs), negs[0].shape)  # 4 elements, each one of shape 10 X 100
        # print(len(seq), seq[0].shape)  # 4 elements, each one of length 100
        # print(len(pos), pos[0].shape)  # 4 elements, each one of length 100
        for ii in range(args.num_neg_seqs):
            neg = [n[ii, :] for n in negs]
            inputs, target = create_combined_dataset_sasrec(u, seq, pos, neg, [args])
            # inputs have ['input_seq', 'positive', 'negative']
            # print(
            #     inputs["input_seq"].shape,
            #     inputs["positive"].shape,
            #     inputs["negative"].shape,
            # )
            # sys.exit()
            pos_logits, neg_logits, loss_mask = model(inputs, training=False)
            print(pos_logits.shape, neg_logits.shape, loss_mask.shape)
            sys.exit()


def evaluate(model, X, y, args):
    if args.model_name == "t-encoder":
        bdim = args.test_batch_size
        num_examples = len(X)
        num_steps, rem = int(num_examples / bdim), int(num_examples % bdim)
        if rem > 0:
            num_steps += 1
        start, end = 0, bdim
        all_scores = []
        preds = []
        for step in tqdm(
            range(num_steps), total=num_steps, ncols=70, leave=False, unit="b"
        ):
            inp, tgt = X[start:end], y[start:end]
            logits = model.predict(inp)
            products = tf.argmax(logits, axis=-1)  # classes
            batch_loss = loss_function(logits, tgt)
            # batch_loss = loss_object(tgt, logits)
            all_scores.append(batch_loss)
            preds.append(products)

            start += bdim
            end += bdim
            if end > num_examples:
                end = num_examples

        preds = tf.concat(preds, axis=0)
        accuracies = tf.equal(y, preds)
        accuracies = tf.cast(accuracies, dtype=tf.float32)
        acc = tf.reduce_mean(accuracies)
        acc = acc.numpy()

    else:
        if args.num_past_sessions == 1:
            num_examples = len(X)
        else:
            num_examples = len(X[0])
        num_steps, rem = int(num_examples / args.batch_size), int(
            num_examples % args.batch_size
        )
        if rem > 0:
            num_steps += 1
        start, end = 0, args.batch_size
        all_scores = []
        for step in tqdm(
            range(num_steps), total=num_steps, ncols=70, leave=False, unit="b"
        ):
            inps, tgt = [x[start:end] for x in X], y[start:end]
            inputs = create_combined_dataset(inps, tgt, [args])
            logits, target_seq, loss_mask = model.predict(inputs)
            pred_seq = tf.math.argmax(logits, axis=-1)  # (b, s)
            pred_seq = pred_seq * tf.cast(loss_mask, dtype=tf.int64)
            pred_seq = pred_seq.numpy()

            batch_score = get_batch_score(target_seq, pred_seq, ignore=[0])
            all_scores.append(batch_score)

            start += args.batch_size
            end += args.batch_size
            if end > num_examples:
                end = num_examples

    return np.mean(all_scores), acc


def get_prediction(model, X, args):
    """
    Returns model prediction on the test data
    """
    if args.model_name == "t-encoder":
        bdim = args.test_batch_size
        num_examples = len(X)
        num_steps, rem = int(num_examples / bdim), int(num_examples % bdim)
        if rem > 0:
            num_steps += 1
        start, end = 0, bdim
        preds = []
        for step in tqdm(
            range(num_steps), total=num_steps, ncols=70, leave=False, unit="b"
        ):
            inp = X[start:end]
            logits = model.predict(inp)
            products = tf.argmax(logits, axis=1)  # classes
            preds.append(products)

            start += bdim
            end += bdim
            if end > num_examples:
                end = num_examples

        preds = tf.concat(preds, axis=0)
        preds = preds.numpy()

    else:
        raise ValueError(f"Unknown model name {args.model_name}")

    return preds


def get_data_shapes(train_X, train_y, val_X, val_y, test_X, test_y, args):
    if args.num_past_sessions == 1:
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        val_X = np.array(val_X)
        val_y = np.array(val_y)
        test_X = np.array(test_X)
        test_y = np.array(test_y)

        print("TRAIN:", train_X.shape, train_y.shape)
        print("VALID:", val_X.shape, val_y.shape)
        print(" TEST:", test_X.shape, test_y.shape)

    elif args.num_past_sessions > 1:
        train_X = [np.array(x) for x in train_X]
        train_y = np.array(train_y)
        val_X = [np.array(x) for x in val_X]
        val_y = np.array(val_y)
        test_X = [np.array(x) for x in test_X]
        test_y = np.array(test_y)

        print([x.shape for x in train_X], train_y.shape)
        print([x.shape for x in val_X], val_y.shape)
        print([x.shape for x in test_X], test_y.shape)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--train_dir", required=True)
parser.add_argument("--colsep", default="\t", type=str)
parser.add_argument("--num_past_sessions", default=2, type=int)
parser.add_argument("--num_target_items", default=1, type=int)
parser.add_argument("--decoder_type", default="lstm", type=str)
parser.add_argument("--mode", default="train", type=str)
parser.add_argument("--predict_proba", default=False, type=bool)

parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--test_batch_size", default=256, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--maxlen", default=100, type=int)
parser.add_argument("--tgt_seq_len", default=12, type=int)
parser.add_argument("--hidden_units", default=50, type=int)
parser.add_argument("--num_blocks", default=2, type=int)
parser.add_argument("--num_epochs", default=41, type=int)
parser.add_argument("--num_heads", default=1, type=int)
parser.add_argument("--dropout_rate", default=0.1, type=float)
parser.add_argument("--l2_emb", default=0.0, type=float)
parser.add_argument("--num_neg_test", default=100, type=int)
parser.add_argument("--num_neg_seqs", default=10, type=int)
parser.add_argument("--model_name", default="sasrec", type=str)
parser.add_argument("--patience", default=5, type=int)

# RNN based
parser.add_argument("--rnn_name", default="gru", type=str)

args = parser.parse_args()
wdir = args.dataset + "_" + args.train_dir
model_path = os.path.join(wdir, "checkpoints/")
checkpoint_dir = model_path  # './reco_training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

result_path = os.path.join(wdir, "result.pkl")

if args.mode == "train":
    if not os.path.isdir(wdir):
        os.makedirs(wdir)
    with open(os.path.join(wdir, "args.txt"), "w") as f:
        f.write(
            "\n".join(
                [
                    str(k) + "," + str(v)
                    for k, v in sorted(vars(args).items(), key=lambda x: x[0])
                ]
            )
        )
    f.close()
    f = open(os.path.join(wdir, "log.txt"), "w")

if args.dataset == "dunhumby":
    data_dir = "/recsys_data/RecSys/dunnhumby_The-Complete-Journey/CSV"
    # train_X, train_y, val_X, val_y, test_X, test_y, item_dict = session_data_partition(
    #     data_dir, args
    # )
    # train_X, train_y, val_X, val_y, test_X, test_y, item_dict = get_session_data(data_dir, args)
    # with open("temp.pkl", "wb") as fw:
    #     pickle.dump((train_X, train_y, val_X, val_y, test_X, test_y, item_dict), fw)

    with open("temp.pkl", "rb") as fr:
        train_X, train_y, val_X, val_y, test_X, test_y, item_dict = pickle.load(fr)

    get_data_shapes(train_X, train_y, val_X, val_y, test_X, test_y, args)
    print(f"Total {len(item_dict)} items in the item-dict")
    if args.num_past_sessions == 1:
        num_examples = len(train_X)
    else:
        num_examples = len(train_X[0])

elif args.dataset.startswith("hnm"):
    data_dir = "/recsys_data/RecSys/h_and_m_personalized_fashion_recommendation"
    # train_X, train_y, val_X, val_y, test_X, usernum, itemnum = create_dataset(
    #     data_dir, args
    # )
    # with open(os.path.join(data_dir, args.dataset + ".pkl"), "wb") as fw:
    #     pickle.dump((train_X, train_y, val_X, val_y, test_X, usernum, itemnum), fw)
    # print(f"Training data written in {args.dataset + '.pkl'}")

    with open(os.path.join(data_dir, args.dataset + ".pkl"), "rb") as fr:
        train_X, train_y, val_X, val_y, test_X, usernum, itemnum = pickle.load(fr)
    print(f"Total {usernum} users and {itemnum} items")
    print("TRAIN:", train_X.shape, train_y.shape)
    print("VALID:", val_X.shape, val_y.shape)
    print(" TEST:", test_X.shape)
    num_examples = len(train_X)

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")

if args.model_name == "sasrec":
    train_sampler = WarpSampler(
        train_X,
        train_y,
        item_dict,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        n_workers=3,
    )
    valid_sampler = WarpSampler(
        val_X,
        val_y,
        item_dict,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        neg_examples=args.num_neg_seqs,
        n_workers=3,
    )
    test_sampler = WarpSampler(
        test_X,
        test_y,
        item_dict,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        neg_examples=args.num_neg_seqs,
        n_workers=3,
    )
    # base input signatures
    train_step_signature = [
        {
            "users": tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
            "input_seq": tf.TensorSpec(shape=(None, args.maxlen), dtype=tf.int64),
            "positive": tf.TensorSpec(shape=(None, args.maxlen), dtype=tf.int64),
            "negative": tf.TensorSpec(shape=(None, args.maxlen), dtype=tf.int64),
        },
        tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
    ]
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    print("Invoking vanilla SASREC model ... ")
    f.write("Invoking vanilla SASREC model ... ")
    model = SASREC(
        item_num=len(item_dict),
        seq_max_len=args.maxlen,
        num_blocks=args.num_blocks,
        embedding_dim=args.hidden_units,
        attention_dim=args.hidden_units,
        attention_num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        conv_dims=[args.hidden_units, args.hidden_units],
        l2_reg=args.l2_emb,
        num_neg_test=args.num_neg_test,
    )

    def loss_function(pos_logits, neg_logits, istarget):
        pos_logits = pos_logits[:, 0]
        neg_logits = neg_logits[:, 0]

        # for logits
        loss = tf.reduce_sum(
            -tf.math.log(tf.math.sigmoid(pos_logits) + 1e-24) * istarget
            - tf.math.log(1 - tf.math.sigmoid(neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)

        # for probabilities
        # loss = tf.reduce_sum(
        #         - tf.math.log(pos_logits + 1e-24) * istarget -
        #         tf.math.log(1 - neg_logits + 1e-24) * istarget
        # ) / tf.reduce_sum(istarget)
        reg_loss = tf.compat.v1.losses.get_regularization_loss()
        # reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        # loss += sum(reg_losses)
        loss += reg_loss
        return loss

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        with tf.GradientTape() as tape:
            pos_logits, neg_logits, loss_mask = model(inp, training=True)
            loss = loss_function(pos_logits, neg_logits, loss_mask)
            # loss = loss_function_(tar, predictions)
            # loss = model.loss_function(*predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        # train_accuracy(accuracy_function(tar, predictions))
        return loss


elif args.model_name == "sasrec2":
    # base input signatures
    if args.num_past_sessions == 1:
        train_step_signature = [
            {
                "input_seq1": tf.TensorSpec(shape=(None, args.maxlen), dtype=tf.int64),
                "input_seq2": tf.TensorSpec(shape=(None, args.maxlen), dtype=tf.int64),
                "target_seq": tf.TensorSpec(shape=(None, args.maxlen), dtype=tf.int64),
            },
        ]
    elif args.num_past_sessions == 2:
        train_step_signature = [
            {
                "input_seq1": tf.TensorSpec(shape=(None, args.maxlen), dtype=tf.int64),
                "input_seq2": tf.TensorSpec(shape=(None, args.maxlen), dtype=tf.int64),
                "input_seq3": tf.TensorSpec(shape=(None, args.maxlen), dtype=tf.int64),
                "target_seq": tf.TensorSpec(shape=(None, args.maxlen), dtype=tf.int64),
            },
        ]
    elif args.num_past_sessions == 3:
        train_step_signature = [
            {
                "input_seq1": tf.TensorSpec(shape=(None, args.maxlen), dtype=tf.int64),
                "input_seq2": tf.TensorSpec(shape=(None, args.maxlen), dtype=tf.int64),
                "input_seq3": tf.TensorSpec(shape=(None, args.maxlen), dtype=tf.int64),
                "input_seq4": tf.TensorSpec(shape=(None, args.maxlen), dtype=tf.int64),
                "target_seq": tf.TensorSpec(shape=(None, args.maxlen), dtype=tf.int64),
            },
        ]
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )

    def loss_function(logits, labels, loss_mask):
        # logits: (b, s, V)
        # labels: (b, s)
        # mask: (b, s)

        loss = loss_object(labels, logits)
        loss = tf.reduce_sum(loss * loss_mask)

        reg_loss = tf.compat.v1.losses.get_regularization_loss()
        loss += reg_loss
        return loss

    print("Invoking Transformer SASREC model ... ")
    f.write("Invoking Transformer SASREC model ... ")
    model = SASREC2(
        item_num=len(item_dict),
        seq_max_len=args.maxlen,
        num_blocks=args.num_blocks,
        embedding_dim=args.hidden_units,
        attention_dim=args.hidden_units,
        attention_num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        conv_dims=[args.hidden_units, args.hidden_units],
        l2_reg=args.l2_emb,
        num_past_sessions=args.num_past_sessions,
    )

    @tf.function(input_signature=train_step_signature)
    def train_step(inp):
        with tf.GradientTape() as tape:
            logits, target, loss_mask = model(inp, training=True)
            loss = loss_function(logits, target, loss_mask)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        # train_accuracy(accuracy_function(tar, predictions))
        return loss

    metric_names = ["Jaccard"]

elif args.model_name == "t-encoder":
    print("Invoking Transformer Encoder model ... ")
    if args.mode == "train":
        f.write("Invoking Transformer Encoder model ... ")
    model = TENCODER(
        item_num=itemnum,
        seq_max_len=args.maxlen,
        tgt_seq_len=args.tgt_seq_len,
        num_blocks=args.num_blocks,
        embedding_dim=args.hidden_units,
        attention_dim=args.hidden_units,
        attention_num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        conv_dims=[args.hidden_units, args.hidden_units],
        l2_reg=args.l2_emb,
        predict_proba=args.predict_proba,
    )
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=not args.predict_proba,
        reduction="none",  # reduction=tf.keras.losses.Reduction.AUTO
    )

    def loss_function(logits, labels):
        loss = loss_object(y_true=labels, y_pred=logits)
        mask = tf.logical_not(
            tf.math.equal(labels, 0)
        )  # output 0 for y=0 else output 1
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = mask * loss
        loss = tf.reduce_mean(loss)
        reg_loss = tf.compat.v1.losses.get_regularization_loss()
        loss += reg_loss
        return loss

    train_step_signature = [
        tf.TensorSpec(shape=(None, args.maxlen), dtype=tf.int64),
        tf.TensorSpec(shape=(None, args.tgt_seq_len), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, target):
        with tf.GradientTape() as tape:
            logits = model(inp, training=True)
            loss = loss_function(logits, target)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        # train_accuracy(accuracy_function(tar, predictions))
        return loss

    metric_names = ["scc", "accuracy"]  # sparse-categorical-crossentropy

else:
    raise ValueError(f"Unknown model name {args.model_name}")

extra_argument = [args]
optimizer = tf.keras.optimizers.Adam(
    learning_rate=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7
)


def accuracy_function(real, pred):
    pred_class = tf.where(pred > 0.5, 1, 0)
    accuracies = tf.equal(real, pred_class)
    # accuracies = tf.equal(tf.argmax(real, axis=1), tf.argmax(pred, axis=1))
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    return tf.reduce_mean(accuracies)


T = 0.0
t0 = time.time()
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

if args.mode == "test":
    # model = keras.models.load_model(model_path)
    # model.load_weights(model_path)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    test_pred = get_prediction(model, test_X, args)
    with open(result_path, "wb") as fw:
        pickle.dump((test_pred), fw)
    print(f"Test predictions are written in {result_path}")


elif args.mode == "train":
    best_ndcg = np.inf
    best_model = None
    patience = 0
    num_steps = int(num_examples / args.batch_size)
    rem = int(num_examples % args.batch_size)
    if rem > 0:
        num_steps += 1

    for epoch in range(1, args.num_epochs + 1):

        step_loss = []
        train_loss.reset_states()
        start, end = 0, args.batch_size
        for step in tqdm(
            range(num_steps), total=num_steps, ncols=70, leave=False, unit="b"
        ):

            if args.model_name == "sasrec":
                u, seq, pos, neg = train_sampler.next_batch()
                inputs, target = create_combined_dataset_sasrec(
                    u, seq, pos, neg, extra_argument
                )
                loss = train_step(inputs, target)
                t_test = evaluate_sasrec(model, test_sampler, len(test_X), args)

            elif args.model_name == "sasrec2":
                if args.num_past_sessions == 1:
                    inps, tgt = train_X[start:end], train_y[start:end]
                else:
                    inps, tgt = [x[start:end] for x in train_X], train_y[start:end]

                inputs = create_combined_dataset(inps, tgt, extra_argument)
                loss = train_step(inputs)

            elif args.model_name == "t-encoder":
                inp, tgt = train_X[start:end], train_y[start:end]
                # logits = model(inp, training=True)
                # print(inp.shape, tgt.shape, logits.shape)
                loss = train_step(inp, tgt)
                # t_valid = evaluate(model, val_X, val_y, args)
                # test_pred = get_prediction(model, test_X, args)

            step_loss.append(loss)
            start += args.batch_size
            end += args.batch_size
            if end > num_examples:
                end = num_examples

        # model.save_weights(model_path)
        # temp1 = evaluate(model, train_X, train_y, args)
        val_perf = evaluate(model, val_X, val_y, args)
        # print(temp1, temp2)
        # sys.exit()

        print(f"Epoch: {epoch}, Loss: {np.mean(step_loss):.3f}, {val_perf[0]:.3f}")
        # print(
        #     f"Epoch: {epoch}, Train Loss: {np.mean(step_loss):.3f}, {train_loss.result():.3f}"
        # )
        f.write(
            f"Epoch: {epoch}, Train Loss: {np.mean(step_loss):.3f}, {train_loss.result():.3f}\n"
        )
        if epoch % 5 == 0:
            t1 = time.time() - t0
            T += t1
            print("Evaluating...")
            t_valid = evaluate(model, val_X, val_y, args)
            if args.model_name == "t-encoder":
                print(
                    f"epoch: {epoch}, time: {T}, valid-{metric_names[0]}: {t_valid[0]:.4f}, valid-{metric_names[1]}: {t_valid[1]:.4f}"
                )

            else:
                t_test = evaluate(model, test_X, test_y, args)
                print(
                    f"epoch: {epoch}, time: {T}, valid-{metric_names[0]}: {t_valid[0]:.4f}, test-{metric_names[0]}: {t_test[0]:.4f})"
                )

            if t_valid[0] < best_ndcg:
                print("Performance improved ... updated the model.")
                best_ndcg = t_valid[0]
                checkpoint.save(file_prefix=checkpoint_prefix)

                # best_model = model
                # best_model.save(model_path, save_format="tf")
                # best_model.save_weights(model_path)
            else:
                patience += 1
                if patience == args.patience:
                    print(f"Maximum patience {patience} reached ... exiting!")

            f.write("validation: " + str(t_valid[0]) + "\n")
            f.flush()
            t0 = time.time()

    # Final validation run
    t_valid = evaluate(model, val_X, val_y, args)

    if t_valid[0] < best_ndcg:
        print("Performance improved ... updated the model.")
        best_ndcg = t_valid[0]
        checkpoint.save(file_prefix=checkpoint_prefix)
        # best_model = model
        # best_model.save(model_path, save_format="tf")

    if args.model_name == "t-encoder":
        print(
            f"epoch: {epoch}, time: {T}, valid-{metric_names[0]}: {t_valid[0]:.4f}, valid-{metric_names[1]}: {t_valid[1]:.4f}"
        )
        f.write(
            f"\nepoch: {epoch}, time: {T}, valid-{metric_names[0]}: {t_valid[0]:.4f}, valid-{metric_names[1]}: {t_valid[1]:.4f}"
        )

    else:
        t_test = evaluate(model, test_X, test_y, args)
        print(
            f"epoch: {epoch}, time: {T}, valid-{metric_names[0]}: {t_valid[0]:.4f}, test-{metric_names[0]}: {t_test[0]:.4f})"
        )
        f.write(
            f"\nepoch: {epoch}, valid-{metric_names[0]}: {t_valid[0]:.4f}, test-{metric_names[0]}: {t_test[0]:.4f})"
        )

    f.close()

    print("Evaluating test data ...")
    test_pred = get_prediction(model, test_X, args)
    with open(result_path, "wb") as fw:
        pickle.dump((test_pred), fw)
    print(f"Test predictions are written in {result_path}")
    print("TRAINING COMPLETE.")
