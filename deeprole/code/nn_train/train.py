import tensorflow as tf
import numpy as np
import random
import sys

from data import load_data, MATRIX, get_payoff_vector, get_viewpoint_vector

class CFVMaskAndAdjustLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(CFVMaskAndAdjustLayer, self).__init__(*args, **kwargs)

    def call(self, inps):
        inp, cfvs = inps

        touched_cfvs = tf.matmul(inp, tf.constant(MATRIX, dtype=tf.float32))
        mask = tf.cast(tf.math.greater(touched_cfvs, tf.constant(0.0)), tf.float32)

        masked_cfvs = tf.math.multiply(cfvs, mask)
        masked_sum = tf.reduce_sum(masked_cfvs)
        mask_sum = tf.reduce_sum(mask)

        return masked_cfvs - masked_sum / mask_sum * mask


class CFVFromWinProbsLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(CFVFromWinProbsLayer, self).__init__(*args, **kwargs)

    def call(self, inps):
        inp_probs = inps[0][:, 5:]
        print "INP_PROBS", inp_probs.shape
        win_probs = inps[1]
        print "WIN_PROBS", win_probs.shape
        factual_values = []
        for player in range(5):
            print "==== player", player, "===="
            good_win_payoff_for_player = tf.constant(get_payoff_vector(player), dtype=tf.float32)
            print "good_win_payoff", good_win_payoff_for_player.shape
            player_payoff = 2.0 * good_win_payoff_for_player * win_probs - good_win_payoff_for_player
            print "player_payoff", player_payoff.shape
            factual_values_for_belief_and_player = inp_probs * player_payoff
            print "factual_values", factual_values_for_belief_and_player.shape

            factual_values_for_player = tf.matmul(factual_values_for_belief_and_player, tf.constant(get_viewpoint_vector(player), dtype=tf.float32))
            print "factual_values_for_player", factual_values_for_player.shape
            factual_values.append(factual_values_for_player)

        result = tf.concat(factual_values, 1)
        print "result", result.shape
        return result




def loss(y_true, y_pred):
    return tf.losses.mean_squared_error(y_true, y_pred) # + tf.losses.huber_loss(y_true, y_pred) #+ tf.losses.mean_pairwise_squared_error(y_true, y_pred)


def create_model_v1():
    inp = tf.keras.layers.Input(shape=(65,))
    x = tf.keras.layers.Dense(128, activation='relu')(inp)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(5*15)(x)
    out = CFVMaskAndAdjustLayer()([inp, x])

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model


def create_model_v2():
    inp = tf.keras.layers.Input(shape=(65,))
    x = tf.keras.layers.Dense(80, activation='relu')(inp)
    x = tf.keras.layers.Dense(80, activation='relu')(x)
    win_probs = tf.keras.layers.Dense(60, activation='sigmoid')(x)
    out = CFVFromWinProbsLayer()([inp, win_probs])

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model


VAL_DATAPOINTS = 20000
def train(num_succeeds, num_fails, propose_count):
    print "Loading data..."
    _, X, Y = load_data(num_succeeds, num_fails, propose_count)
    idx = np.random.permutation(len(X))
    X = X[idx]
    Y = Y[idx]

    X_val, X_train = X[:VAL_DATAPOINTS], X[VAL_DATAPOINTS:]
    Y_val, Y_train = Y[:VAL_DATAPOINTS], Y[VAL_DATAPOINTS:]

    for num_datapoints in [20000, 40000, 60000, 80000, 100000]:
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            'models/{}.h5'.format(num_datapoints),
            save_best_only=True
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir='logs/{}'.format(num_datapoints)
        )

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=int(100 * (100000.0/num_datapoints)),
            verbose=True
        )

        print "Fitting..."
        model = create_model_v1()
        model.fit(
            x=X_train[:num_datapoints],
            y=Y_train[:num_datapoints],
            batch_size=4000,
            epochs=int(5000 * (100000.0/num_datapoints)),
            validation_data=(X_val, Y_val),
            callbacks=[checkpoint_callback, tensorboard_callback, early_stopping_callback]
        )
    return


def compare(a, b):
    a = np.abs(a)
    b = np.abs(b)
    m = max(np.max(a), np.max(b))
    increment = m/20
    print "{: <20}{: >20}".format('A', 'B')
    for i in range(len(a)):
        print "{: <20}{: >20}".format('#' * int(a[i]/increment), '#' * int(b[i]/increment))


def random_compare(X, Y, model):
    index = int(len(X) * random.random())
    a = Y[index]
    b = model.predict(np.array([X[index]]))[0]
    compare(a, b)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Usage:"
        print "python train.py <num_succeeds> <num_fails> <propose_count>"
    _, num_succeeds, num_fails, propose_count = sys.argv
    train(num_succeeds, num_fails, propose_count)
