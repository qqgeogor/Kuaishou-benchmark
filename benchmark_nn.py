
# coding: utf-8

# In[1]:




# In[1]:

import numpy as np
import scipy as sp
from scipy import sparse as ssp
from scipy.stats import spearmanr
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler

from config import path


# In[2]:

columns = ['user_id','photo_id','click','like','follow','time','playing_time','duration_time']
train_interaction = pd.read_table(path+'train_interaction.txt',header=None)
train_interaction.columns = columns
test_columns = ['user_id','photo_id','time','duration_time']

test_interaction = pd.read_table(path+'test_interaction.txt',header=None)
test_interaction.columns = test_columns


# In[2]:

data = pd.concat([train_interaction,test_interaction])


# In[3]:

from sklearn.preprocessing import LabelEncoder
le_user = LabelEncoder()
data['user_id'] = le_user.fit_transform(data['user_id'])

le_photo = LabelEncoder()

data['photo_id'] = le_photo.fit_transform(data['photo_id'])


# In[25]:


def generate_doc(df,name,concat_name):
    res = df.astype(str).groupby(name)[concat_name].apply((lambda x :' '.join(x))).reset_index()
    res.columns = [name,'%s_doc'%concat_name]
    return res

user_doc = generate_doc(data,'photo_id','user_id')
photo_doc = generate_doc(data,'user_id','photo_id')


# In[26]:

user_doc.to_csv(path+'user_doc.csv',index=False)
photo_doc.to_csv(path+'photo_doc.csv',index=False)
 


# In[17]:

user_doc['user_id_doc'].to_csv(path+'user.adjlist',index=False)
photo_doc['photo_id_doc'].to_csv(path+'photo.adjlist',index=False)


# In[18]:


# photo_count = data.groupby(['photo_id'])['user_id'].count()


# In[19]:

# %matplotlib inline
# print(photo_count.describe())
# photo_count.hist()


# In[20]:

import commands
commands.getoutput("bash train_deepwalk.sh")


# In[4]:

def read_emb(path):
    count=0
    f = open(path,'r')
    emb_dict = dict()
    for line in f:
        if count==0:
            count+=1
            continue
        line = line.split(' ')
        id = int(line[0])
        
        weights = line[1:]
        weights = np.array([float(i) for i in weights])
        count+=1
        emb_dict[id] = weights
    return emb_dict


# In[5]:

user_emb = read_emb(path+'user.emb')
photo_emb = read_emb(path+'photo.emb')


# In[6]:

print('Preparing embedding matrix user')
EMBEDDING_DIM_USER=64
nb_users = data['user_id'].nunique()
embedding_matrix_user = np.zeros((nb_users, EMBEDDING_DIM_USER))
print(embedding_matrix_user.shape)
for word in user_emb.keys():
    embedding_vector = user_emb.get(word)
    embedding_matrix_user[word] = embedding_vector


# In[7]:

print('Null word embeddings user: %d' % np.sum(np.sum(embedding_matrix_user, axis=1) == 0))


# In[8]:

print('Preparing embedding matrix photo')
EMBEDDING_DIM_PHOTO=64
nb_photos = data['photo_id'].nunique()
embedding_matrix_photo = np.zeros((nb_photos, EMBEDDING_DIM_PHOTO))
print(embedding_matrix_photo.shape)
for word in photo_emb.keys():
    embedding_vector = photo_emb.get(word)
    embedding_matrix_photo[word] = embedding_vector
    


# In[9]:

print('Null word embeddings photo: %d' % np.sum(np.sum(embedding_matrix_photo, axis=1) == 0))


# In[10]:


########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation


from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GRU, Embedding,SpatialDropout1D, Dropout, Activation,Bidirectional,TimeDistributed,CuDNNGRU
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop,Adam
from keras.layers import GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.layers import average,dot,maximum,multiply,add



# In[11]:


########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer   
from string import punctuation
from sklearn.metrics import roc_auc_score

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,Conv1D,AveragePooling1D,MaxPooling1D,Flatten,merge,TimeDistributed,ZeroPadding1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler,Callback
from keras.layers import GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.layers import average,dot,maximum,multiply,add

class AucCallback(Callback):  #inherits from Callback
    
    def __init__(self, validation_data=(), patience=25,is_regression=False,best_model_name='best_keras.mdl',feval='roc_auc_score',batch_size=128):
        super(Callback, self).__init__()
        
        self.patience = patience
        self.X_test, self.y_test = validation_data  #tuple of validation X and y
        self.best = -np.inf
        self.wait = 0  #counter for patience
        self.best_model=None
        self.best_model_name = best_model_name
        self.is_regression = is_regression
        self.y_test = self.y_test#.astype(np.int)
        self.feval = feval
        self.batch_size = batch_size
    def on_epoch_end(self, epoch, logs={}):
        # p = self.model.predict(self.X_test,batch_size=self.batch_size, verbose=0)#.ravel()
        p = []
        # for X_batch,y_batch in test_batch_generator(self.X_test,self.y_test,batch_size=self.batch_size):
        #     p.append(model.predict(X_batch,batch_size=batch_size))
        # p = np.concatenate(p).ravel()
        p = model.predict(self.X_test)

        current = 0.0
        if self.feval=='roc_auc_score':

            current+= roc_auc_score(self.y_test.ravel(),p.ravel())


        if current > self.best:
            self.best = current
            self.wait = 0
            self.model.save_weights(self.best_model_name,overwrite=True)
            

        else:
            if self.wait >= self.patience:
                self.model.stop_training = True
                print('Epoch %05d: early stopping' % (epoch))
                
                
            self.wait += 1 #incremental the number of times without improvement
        print('Epoch %d Auc: %f | Best Auc: %f \n' % (epoch,current,self.best))


# In[12]:

act = 'relu'


# In[13]:

########################################
## define the model structure
########################################
embedding_layer_user = Embedding(nb_users,
        EMBEDDING_DIM_USER,
        weights=[embedding_matrix_user],
        trainable=False)

embedding_layer_user2 = Embedding(nb_users,
        EMBEDDING_DIM_USER,
        trainable=True)

embedding_layer_photo = Embedding(nb_photos,
        EMBEDDING_DIM_PHOTO,
        weights=[embedding_matrix_photo],
        trainable=False)


# In[60]:

MAX_SENTENCE_LENGTH = 30


# In[61]:


input_user = Input(shape=(1,), dtype='int32')
input_photo = Input(shape=(1,), dtype='int32')
input_user_mean = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')

embedded_user= embedding_layer_user(input_user)
embedded_user2= embedding_layer_user2(input_user)
embedded_photo= embedding_layer_photo(input_photo)


embedded_user2_agg = embedding_layer_user2(input_user_mean)
embedded_user_agg = embedding_layer_user(input_user_mean)



embedded_user = Flatten()(embedded_user)
embedded_user2 = Flatten()(embedded_user2)
embedded_photo = Flatten()(embedded_photo)
embedded_user2_mean = GlobalAveragePooling1D()(embedded_user2_agg)
embedded_user2_max = GlobalMaxPooling1D()(embedded_user2_agg)
embedded_user_max = GlobalMaxPooling1D()(embedded_user_agg)


flatten_list = [
    embedded_user,
    embedded_user2,
    embedded_photo,
#     embedded_user2_mean,
    embedded_user2_max,
#     embedded_user_max,
]


# In[62]:

merged = concatenate(flatten_list,name='match_concat')
merged = Dense(128, activation=act)(merged)
merged = BatchNormalization()(merged)
merged = Dropout(0.25)(merged)

preds = Dense(1, activation='sigmoid')(merged)

########################################
## train the model
########################################
model = Model(inputs=[input_user,input_photo,input_user_mean ],         outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
print(model.summary())


# In[16]:

len_train = train_interaction.shape[0]
train = data[:len_train]
test = data[len_train:]


# In[28]:

user_doc['photo_id'] = user_doc['photo_id'].astype(int)
user_doc['user_id_doc'] = user_doc['user_id_doc'].apply(lambda x:[int(s) for s in x.split(' ')])




# In[29]:

train = pd.merge(train,user_doc,on='photo_id',how='left')
test = pd.merge(test,user_doc,on='photo_id',how='left')



# In[17]:

y = train_interaction['click'].values
skf =StratifiedKFold(n_splits=5, shuffle=True, random_state=1).split(train_interaction['user_id'],y)
for ind_tr, ind_te in skf:
    break
    


# In[63]:

train_user_mean = pad_sequences(train['user_id_doc'].values, maxlen=MAX_SENTENCE_LENGTH)
test_user_mean = pad_sequences(test['user_id_doc'].values, maxlen=MAX_SENTENCE_LENGTH)
print('Shape of train_user_mean tensor:', train_user_mean.shape)
print('Shape of test_user_mean tensor:', test_user_mean.shape)


# In[ ]:

X_train = [
    train['user_id'].values[ind_tr],
    train['photo_id'].values[ind_tr],
    train_user_mean[ind_tr]
]
X_test = [
    train['user_id'].values[ind_te],
    train['photo_id'].values[ind_te],
    train_user_mean[ind_te]
]
X_t = [
    test['user_id'].values,
    test['photo_id'].values,
    test_user_mean
]

y_train = y[ind_tr]
y_test = y[ind_te]


# In[ ]:

STAMP = 'base'
print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=5)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

batch_size = 1024

def schedule(index):
    if index<10:
        return 0.001
    else:
        return 0.0001


lrs = LearningRateScheduler(schedule)
auc_callback = AucCallback(validation_data=(X_test,y_test),patience=5,best_model_name=bst_model_path,batch_size=batch_size)
callbacks = [
    # model_checkpoint,
    # early_stopping,
    # lrs,
    auc_callback,
    ]



hist = model.fit(X_train, y_train,         validation_data=(X_test, y_test),         epochs=50, batch_size=batch_size, shuffle=True,          callbacks=callbacks)


# In[ ]:

model.load_weights(bst_model_path)

y_pred = model.predict(X_test)
score = roc_auc_score(y_test,y_pred)
print('auc score:%s'%score)

from sklearn.metrics import log_loss
score = log_loss(y_test,y_pred)
print('logloss score:%s'%score)

y_sub = model.predict(X_t)

submission = pd.DataFrame()
submission['user_id'] = test_interaction['user_id']
submission['photo_id'] = test_interaction['photo_id']
submission['click_probability'] = y_sub
submission['click_probability'].apply(lambda x:float('%.6f' % x))
submission.to_csv('submission_nn.txt',sep='\t',index=False,header=False)



