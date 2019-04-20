import tensorflow as tf
import numpy as np
import random
import sys

from data import load_data, MATRIX

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


def loss(y_true, y_pred):
    return tf.losses.mean_squared_error(y_true, y_pred) # + tf.losses.huber_loss(y_true, y_pred) #+ tf.losses.mean_pairwise_squared_error(y_true, y_pred)


def create_model():
    inp = tf.keras.layers.Input(shape=(65,))
    x = tf.keras.layers.Dense(128, activation='relu')(inp)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(5*15)(x)
    out = CFVMaskAndAdjustLayer()([inp, x])

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss=loss, metrics=['mse'])
    return model

def train(num_succeeds, num_fails, propose_count):
    model = create_model()

    print "Loading data..."
    _, X, Y = load_data(num_succeeds, num_fails, propose_count)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        'models/{}_{}_{}.h5'.format(num_succeeds, num_fails, propose_count),
        save_best_only=True
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs/{}_{}_{}'.format(num_succeeds, num_fails, propose_count))

    print "Fitting..."
    model.fit(
        x=X,
        y=Y,
        batch_size=4096,
        epochs=3000,
        validation_split=0.1,
        callbacks=[checkpoint_callback, tensorboard_callback]
    )
    return X, Y, model


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
