#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import keras
from keras import regularizers
from keras.layers import *
from keras.callbacks import *
from keras.models import Model, Sequential, load_model
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.constraints import max_norm
import pandas as pd
import numpy as np
import utils
from Attention_keras import *
import lambdapre

COMBINEFL = [
    'stacktrace', 'slicing_count', 'predicateswitching',
    'slicing_intersection', 'metallaxis', 'ochiai', 'muse', 'slicing', 'dstar'
]
#COMBINEFL = ['stacktrace', 'slicing_count', 'predicateswitching', 'slicing_intersection', 'metallaxis',  'muse', 'slicing']

FORMULAS = [
    'tarantula', 'ochiai', 'barinel', 'dstar2', 'jaccard', 'er1a', 'er1b',
    'gp2', 'gp3', 'gp13', 'gp19'
]
'''FORMULAS = [
    'tarantula', 'ochiai', 'barinel', 'dstar2', 'jaccard', 'er1a',
    'er1b', 'er5a', 'er5b', 'er5c', 'gp2', 'gp3', 'gp13', 'gp19'
]'''
SPECTRUM = ['ep', 'ef', 'np', 'nf']
CODE = ['code']
METRICS1 = ['level', 'nLines', 'nComments', 'nTokens', 'nChars']
vocab_size = 2000
embedding_size = 16
max_length = 20


def sAtt(att_dim, inputs, name):
    V = inputs
    QK = Dense(att_dim, use_bias=None)(inputs)
    QK = Activation("softmax", name=name)(QK)
    MV = Multiply()([V, QK])
    return (MV)


def create_base_network(input_dim, isconcat=False):
    sbfl_input = Input(shape=(input_dim, ))
    combine_input = sbfl_input
    if isconcat:
        code_input = Input(shape=(None, ))
        combine_input = [code_input, sbfl_input]

        output = Embedding(vocab_size, embedding_size,
                           input_length=max_length)(code_input)

        embeddings = Position_Embedding()(output)
        output = Attention(8, 16)([embeddings, embeddings, embeddings])
        #output = GlobalAveragePooling1D()(output)
        output = GlobalMaxPooling1D()(output)  #average change to max
        #output = Dense(8, activation='relu')(output)
        output = Dense(1, activation='sigmoid')(output)
        #output = LSTM(1, kernel_regularizer=regularizers.l2(0.01))(output)
        #output = Activation('sigmoid')(output)
        sbfl_input = concatenate([output, sbfl_input])
        atts1 = sAtt(input_dim + 1, sbfl_input,
                     "attention_vec1")  #+1= output vector dim
    else:
        atts1 = sAtt(input_dim, sbfl_input, "attention_vec1")

    output = Dense(16, activation='relu')(atts1)
    output = BatchNormalization()(output)
    output = Dense(8, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dense(1)(output)
    return Model(inputs=combine_input, outputs=output)


def create_meta_network(input_dim, base_network, isconcat=False):
    input_a = Input(shape=(input_dim, ))
    input_b = Input(shape=(input_dim, ))
    combine_input = [input_a, input_b]
    if isconcat:
        input_a = [Input(shape=(None, )), input_a]
        input_b = [Input(shape=(None, )), input_b]
        combine_input = input_a + input_b

    rel_score = base_network(input_a)
    irr_score = base_network(input_b)
    diff = Subtract()([rel_score, irr_score])
    prob = Activation('sigmoid')(diff)

    return Model(inputs=combine_input, outputs=prob)


def train(n, isconcat=False, modelname='ranknet-concat-model.h5'):
    # ochiai is changed to ochiai_x and ochiai_y while merging them.
    sbfl = [
        formula if formula != 'ochiai' else formula + '_y'
        for formula in FORMULAS
    ]
    combinefl = [com if com != 'ochiai' else com + '_x' for com in COMBINEFL]
    #keys = combinefl
    keys = SPECTRUM + sbfl + combinefl + METRICS1

    tokenizer = pickle.load(open('data/tokenizer.pickle', 'rb'))
    for i in range(n):
        data_dir = 'data/cross_datasvm/{}'.format(i)
        train_df = pd.read_csv(
            '{}/train_pairs.csv'.format(data_dir)).sample(frac=1)
        test_df = pd.read_csv(
            '{}/test_pairs.csv'.format(data_dir)).sample(frac=1)

        features1 = [feature + '_1' for feature in keys]
        features2 = [feature + '_2' for feature in keys]
        train_features1 = train_df[features1]
        train_features2 = train_df[features2]
        test_features1 = test_df[features1]
        test_features2 = test_df[features2]

        X_train = [train_features1, train_features2]
        X_test = [test_features1, test_features2]

        y_train = train_df['label'].values
        y_test = test_df['label'].values

        if isconcat:
            train_code1 = pad_sequences(tokenizer.texts_to_sequences(
                train_df['code_1'].copy()),
                                        maxlen=max_length)
            train_code2 = pad_sequences(tokenizer.texts_to_sequences(
                train_df['code_2'].copy()),
                                        maxlen=max_length)
            test_code1 = pad_sequences(tokenizer.texts_to_sequences(
                test_df['code_1'].copy()),
                                       maxlen=max_length)
            test_code2 = pad_sequences(tokenizer.texts_to_sequences(
                test_df['code_2'].copy()),
                                       maxlen=max_length)

            X_train = [
                train_code1, train_features1, train_code2, train_features2
            ]
            X_test = [test_code1, test_features1, test_code2, test_features2]

        INPUT_DIM = int(len(keys))
        base_network = create_base_network(INPUT_DIM, isconcat)
        model = create_meta_network(INPUT_DIM, base_network, isconcat)
        opt = keras.optimizers.Adam(lr=0.1, decay=0.1)
        model.compile(optimizer=opt,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss',
                           patience=5,
                           verbose=1,
                           restore_best_weights=True)
        #history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=4096, epochs=6, verbose=1, callbacks=[es])
        model.fit(X_train,
                  y_train,
                  validation_data=(X_test, y_test),
                  batch_size=512,
                  epochs=8,
                  verbose=1,
                  callbacks=[es])
        base_network.save('{}/'.format(data_dir) + modelname)
        '''plt.cla()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig('{}/loss.png'.format(data_dir))'''


def train_lambda(n, isconcat=False, modelname='lambdanet-concat-model.h5'):
    sbfl = [
        formula if formula != 'ochiai' else formula + '_y'
        for formula in FORMULAS
    ]
    combinefl = [com if com != 'ochiai' else com + '_x' for com in COMBINEFL]
    #keys = sbfl+ METRICS1+SPECTRUM
    #keys = combinefl
    keys = sbfl + combinefl + METRICS1 + SPECTRUM
    allfea = keys + ['code']

    tokenizer = pickle.load(open('data/tokenizer.pickle', 'rb'))
    for i in range(n):
        #data_dir = 'data/cross_data/{}'.format(i)
        data_dir = 'data/cross_datasvm/{}'.format(i)
        df = pd.read_csv('{}/train.csv'.format(data_dir))

        test_df = pd.read_csv(
            '{}/test_pairs.csv'.format(data_dir)).sample(frac=1)
        y_test = test_df['label'].values
        features1 = [feature + '_1' for feature in keys]
        features2 = [feature + '_2' for feature in keys]
        test_features1 = test_df[features1]
        test_features2 = test_df[features2]

        X = df[allfea].values
        y = df['faulty'].values
        qid = df['qid'].values
        # add weight to train data pairs
        X1_trans, X2_trans, y_trans, weight = lambdapre.transform_pairwise(
            X, y, qid)

        train_features1 = X1_trans[:, :-1]
        train_features2 = X2_trans[:, :-1]
        X_train = [train_features1, train_features2]
        y_train = y_trans
        if isconcat:
            train_code1 = pad_sequences(tokenizer.texts_to_sequences(
                X1_trans[:, -1]),
                                        maxlen=max_length)
            train_code2 = pad_sequences(tokenizer.texts_to_sequences(
                X2_trans[:, -1]),
                                        maxlen=max_length)
            X_train = [
                train_code1, train_features1, train_code2, train_features2
            ]

            test_code1 = pad_sequences(tokenizer.texts_to_sequences(
                test_df['code_1'].copy()),
                                       maxlen=max_length)
            test_code2 = pad_sequences(tokenizer.texts_to_sequences(
                test_df['code_2'].copy()),
                                       maxlen=max_length)
            X_test = [test_code1, test_features1, test_code2, test_features2]

        INPUT_DIM = int(len(keys))
        base_network = create_base_network(INPUT_DIM, isconcat)
        model = create_meta_network(INPUT_DIM, base_network, isconcat)
        '''model = load_model(filepath='{}/'.format(data_dir) + 'lambdanet-b512-model.h5',
                           custom_objects={'Position_Embedding': Position_Embedding, 'Attention': Attention})
'''
        opt = keras.optimizers.Adam(lr=0.1, decay=0.2)
        model.compile(optimizer=opt,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        es = EarlyStopping(monitor='val_loss',
                           patience=10,
                           verbose=1,
                           restore_best_weights=True)
        #es = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        #best_weights_filepath = '{}/'.format(data_dir) + 'best_weights.hdf5'
        '''saveBestModel = ModelCheckpoint(best_weights_filepath,
                                        monitor='val_loss',
                                        verbose=1,
                                        save_best_only=True,
                                        mode='auto')'''
        #model.fit(X_train, y_train,sample_weight=weight, batch_size=1024, epochs=8, verbose=1,validation_split=0.2,callbacks=[es])
        model.fit(X_train,
                  y_train,
                  sample_weight=weight,
                  validation_data=(X_test, y_test),
                  batch_size=512,
                  epochs=25,
                  verbose=1,
                  callbacks=[es])  #, saveBestModel
        base_network.save('{}/'.format(data_dir) + modelname)


def predict(n,
            isconcat=False,
            modelname='ranknet-model.h5',
            predfilename='ranknet-pred.dat'):
    # ochiai is changed to ochiai_x and ochiai_y while merging them.
    sbfl = [
        formula if formula != 'ochiai' else formula + '_y'
        for formula in FORMULAS
    ]
    combinefl = [com if com != 'ochiai' else com + '_x' for com in COMBINEFL]
    #keys = combinefl
    #keys = sbfl + METRICS1+SPECTRUM
    keys = sbfl + combinefl + METRICS1 + SPECTRUM

    tokenizer = pickle.load(open('data/tokenizer.pickle', 'rb'))
    for i in range(n):
        #data_dir = 'data/cross_data/{}'.format(i)
        data_dir = 'data/cross_datasvm/{}'.format(i)
        test_df = pd.read_csv('{}/test.csv'.format(data_dir))

        test_code = pad_sequences(tokenizer.texts_to_sequences(
            test_df['code'].copy()),
                                  maxlen=max_length)
        test_features = test_df[keys].values

        X_test = test_features
        if isconcat:
            X_test = [np.array(test_code), test_features]
            model = load_model(filepath='{}/'.format(data_dir) + modelname,
                               custom_objects={
                                   'Position_Embedding': Position_Embedding,
                                   'Attention': Attention
                               })
        else:
            model = load_model('{}/'.format(data_dir) + modelname)
        preds = model.predict(X_test)
        np.savetxt('{}/'.format(data_dir) + predfilename, preds, newline='\n')


def main():
    isconcat = True
    #isconcat =False
    #modelname = 'lambdanet-b512-model.h5'
    modelname = 'lambdanet-b512-l01-model.h5'
    #modelname = 'lambdanet-b512-windows-model.h5'

    #predfilename='lamdbanet-b512-pred.dat' #acc1:76 lambdanet max poolingresult
    #predfilename = 'lambdanet-b512-pred.dat'  # acc1:60 lambdanet avg pooling result
    #predfilename = 'rank-pred.dat'#acc1:65 svm result
    predfilename = 'lambdanet-b512-l01-pred.dat'  #acc1:74 lambdanet max poolingresult
    #predfilename = 'lambdanet-b512-windows-pred.dat'

    stage = 3

    if stage <= 0:
        utils.prepare_data(vocab_size)
    if stage <= 1:
        utils.split_data(n=5)
        #utils.split_data(n=10)
    if stage <= 2:
        #train(n=5,isconcat=isconcat,modelname=modelname)
        train_lambda(n=5, isconcat=isconcat, modelname=modelname)
        #train_lambda(n=10, isconcat=isconcat, modelname=modelname)
    if stage <= 3:
        predict(n=5,
                isconcat=isconcat,
                modelname=modelname,
                predfilename=predfilename)
    if stage <= 4:
        utils.calc_metric(n=5, predfilename=predfilename)
        utils.calc_metric_method(n=5, predfilename=predfilename)


if __name__ == "__main__":
    main()
