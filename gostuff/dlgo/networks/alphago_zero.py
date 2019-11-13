from keras.layers import *
from keras.models import Model


def dual_residual_network(input_shape, blocks=20):
    inputs = Input(shape=input_shape)
    first_conv = conv_bn_relu_block(name="init")(inputs)
    res_tower = residual_tower(blocks=blocks)(first_conv)
    policy = policy_head()(res_tower)
    value = value_head()(res_tower)
    return Model(inputs=inputs, outputs=[policy, value])


def dual_conv_network(input_shape, blocks=12):
    inputs = Input(shape=input_shape)
    first_conv = conv_bn_relu_block(name="init")(inputs)
    conv_tower = convolutional_tower(blocks=blocks)(first_conv)
    policy = policy_head()(conv_tower)
    value = value_head()(conv_tower)
    return Model(inputs=inputs, outputs=[policy, value])


def separate_residual_network(input_shape, blocks=20):
    inputs_pol = Input(shape=input_shape)
    first_conv_pol = conv_bn_relu_block(name="init")(inputs_pol)
    res_tower_pol = residual_tower(blocks=blocks)(first_conv_pol)
    policy = policy_head()(res_tower_pol)
    policy_model = Model(inputs=inputs_pol, outputs=policy)

    inputs_val = Input(shape=input_shape)
    first_conv_val = conv_bn_relu_block(name="init")(inputs_val)
    res_tower_val = residual_tower(blocks=blocks)(first_conv_val)
    value = value_head()(res_tower_val)
    value_model = Model(inputs=inputs_val, outputs=value)
    
    return policy_model, value_model


def separate_conv_network(input_shape, blocks=20):
    inputs_pol = Input(shape=input_shape)
    first_conv_pol = conv_bn_relu_block(name="init")(inputs_pol)
    conv_tower_pol = convolutional_tower(blocks=blocks)(first_conv_pol)
    policy = policy_head()(conv_tower_pol)
    policy_model = Model(inputs=inputs_pol, outputs=policy)

    inputs_val = Input(shape=input_shape)
    first_conv_val = conv_bn_relu_block(name="init")(inputs_val)
    conv_tower_val = convolutional_tower(blocks=blocks)(first_conv_val)
    value = value_head()(conv_tower_val)
    value_model = Model(inputs=inputs_val, outputs=value)
    
    return policy_model, value_model


def conv_bn_relu_block(name, activation=True, filters=256, kernel_size=(3, 3),
                       strides=(1, 1), padding="same", init="he_normal"):
    def f(inputs):
        conv = Conv2D(filters=filters, 
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding,
                      kernel_initializer=init,
                      data_format='channels_first',
                      name="{}_conv_block".format(name))(inputs)
        batch_norm = BatchNormalization(axis=1, name="{}_batch_norm".format(name))(conv)
        return Activation("relu", name="{}_relu".format(name))(batch_norm) if activation else batch_norm
    return f    


def residual_block(block_num, **args):
    def f(inputs):
        res = conv_bn_relu_block(name="residual_1_{}".format(block_num), activation=True, **args)(inputs)
        res = conv_bn_relu_block(name="residual_2_{}".format(block_num), activation=False, **args)(res)
        res = add([inputs, res], name="add_{}".format(block_num))
        return Activation("relu", name="{}_relu".format(block_num))(res) 
    return f


def residual_tower(blocks, **args):
    def f(inputs):
        x = inputs
        for i in range(blocks):
            x = residual_block(block_num=i)(x)
        return x
    return f


def convolutional_tower(blocks, **args):
    def f(inputs):
        x = inputs
        for i in range(blocks):
            x = conv_bn_relu_block(name=i)(x)
        return x
    return f


def policy_head():
    def f(inputs):
        conv = Conv2D(filters=2, 
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding="same",
                      name="policy_head_conv_block")(inputs)
        batch_norm = BatchNormalization(axis=1, name="policy_head_batch_norm")(conv)
        activation = Activation("relu", name="policy_head_relu")(batch_norm)
        return Dense(units=19*19+1, name="policy_head_dense")(activation)
    return f    


def value_head():
    def f(inputs):
        conv = Conv2D(filters=1, 
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding="same",
                      name="value_head_conv_block")(inputs)
        batch_norm = BatchNormalization(axis=1, name="value_head_batch_norm")(conv)
        activation = Activation("relu", name="value_head_relu")(batch_norm)
        dense = Dense(units=256, name="value_head_dense", activation="relu")(activation)
        return Dense(units=1, name="value_head_output", activation="tanh")(dense)
    return f
