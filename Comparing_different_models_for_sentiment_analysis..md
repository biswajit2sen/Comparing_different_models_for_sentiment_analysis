```python
#GPU information
!nvidia-smi
```

    Mon May 31 12:17:24 2021       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 465.19.01    Driver Version: 460.32.03    CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla P100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   36C    P0    26W / 250W |      0MiB / 16280MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
    


```python
#Mounting the Google drive
from google.colab import drive
drive.mount('/content/gdrive')
```

    Mounted at /content/gdrive
    


```python
#Changing working directory
%cd /content/gdrive/My Drive/ML-in-colab/Sentiment_analysis_amazon_fine_food_review
```

    /content/gdrive/My Drive/ML-in-colab/Sentiment_analysis_amazon_fine_food_review
    


```python
#Installing Dependencies
!pip install tensorflow_text
```

    Collecting tensorflow_text
    [?25l  Downloading https://files.pythonhosted.org/packages/c0/ed/bbb51e9eccca0c2bfdf9df66e54cdff563b6f32daed9255da9b9a541368f/tensorflow_text-2.5.0-cp37-cp37m-manylinux1_x86_64.whl (4.3MB)
    [K     |████████████████████████████████| 4.3MB 8.3MB/s 
    [?25hRequirement already satisfied: tensorflow-hub>=0.8.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow_text) (0.12.0)
    Requirement already satisfied: tensorflow<2.6,>=2.5.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow_text) (2.5.0)
    Requirement already satisfied: protobuf>=3.8.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow-hub>=0.8.0->tensorflow_text) (3.12.4)
    Requirement already satisfied: numpy>=1.12.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow-hub>=0.8.0->tensorflow_text) (1.19.5)
    Requirement already satisfied: tensorboard~=2.5 in /usr/local/lib/python3.7/dist-packages (from tensorflow<2.6,>=2.5.0->tensorflow_text) (2.5.0)
    Requirement already satisfied: keras-preprocessing~=1.1.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow<2.6,>=2.5.0->tensorflow_text) (1.1.2)
    Requirement already satisfied: termcolor~=1.1.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow<2.6,>=2.5.0->tensorflow_text) (1.1.0)
    Requirement already satisfied: typing-extensions~=3.7.4 in /usr/local/lib/python3.7/dist-packages (from tensorflow<2.6,>=2.5.0->tensorflow_text) (3.7.4.3)
    Requirement already satisfied: astunparse~=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorflow<2.6,>=2.5.0->tensorflow_text) (1.6.3)
    Requirement already satisfied: tensorflow-estimator<2.6.0,>=2.5.0rc0 in /usr/local/lib/python3.7/dist-packages (from tensorflow<2.6,>=2.5.0->tensorflow_text) (2.5.0)
    Requirement already satisfied: flatbuffers~=1.12.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow<2.6,>=2.5.0->tensorflow_text) (1.12)
    Requirement already satisfied: absl-py~=0.10 in /usr/local/lib/python3.7/dist-packages (from tensorflow<2.6,>=2.5.0->tensorflow_text) (0.12.0)
    Requirement already satisfied: wrapt~=1.12.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow<2.6,>=2.5.0->tensorflow_text) (1.12.1)
    Requirement already satisfied: gast==0.4.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow<2.6,>=2.5.0->tensorflow_text) (0.4.0)
    Requirement already satisfied: h5py~=3.1.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow<2.6,>=2.5.0->tensorflow_text) (3.1.0)
    Requirement already satisfied: six~=1.15.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow<2.6,>=2.5.0->tensorflow_text) (1.15.0)
    Requirement already satisfied: google-pasta~=0.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow<2.6,>=2.5.0->tensorflow_text) (0.2.0)
    Requirement already satisfied: opt-einsum~=3.3.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow<2.6,>=2.5.0->tensorflow_text) (3.3.0)
    Requirement already satisfied: wheel~=0.35 in /usr/local/lib/python3.7/dist-packages (from tensorflow<2.6,>=2.5.0->tensorflow_text) (0.36.2)
    Requirement already satisfied: grpcio~=1.34.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow<2.6,>=2.5.0->tensorflow_text) (1.34.1)
    Requirement already satisfied: keras-nightly~=2.5.0.dev in /usr/local/lib/python3.7/dist-packages (from tensorflow<2.6,>=2.5.0->tensorflow_text) (2.5.0.dev2021032900)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-packages (from protobuf>=3.8.0->tensorflow-hub>=0.8.0->tensorflow_text) (56.1.0)
    Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.5->tensorflow<2.6,>=2.5.0->tensorflow_text) (1.0.1)
    Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.5->tensorflow<2.6,>=2.5.0->tensorflow_text) (2.23.0)
    Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.5->tensorflow<2.6,>=2.5.0->tensorflow_text) (0.6.1)
    Requirement already satisfied: google-auth<2,>=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.5->tensorflow<2.6,>=2.5.0->tensorflow_text) (1.30.0)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.5->tensorflow<2.6,>=2.5.0->tensorflow_text) (3.3.4)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.5->tensorflow<2.6,>=2.5.0->tensorflow_text) (1.8.0)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.5->tensorflow<2.6,>=2.5.0->tensorflow_text) (0.4.4)
    Requirement already satisfied: cached-property; python_version < "3.8" in /usr/local/lib/python3.7/dist-packages (from h5py~=3.1.0->tensorflow<2.6,>=2.5.0->tensorflow_text) (1.5.2)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard~=2.5->tensorflow<2.6,>=2.5.0->tensorflow_text) (2020.12.5)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard~=2.5->tensorflow<2.6,>=2.5.0->tensorflow_text) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard~=2.5->tensorflow<2.6,>=2.5.0->tensorflow_text) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard~=2.5->tensorflow<2.6,>=2.5.0->tensorflow_text) (1.24.3)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<2,>=1.6.3->tensorboard~=2.5->tensorflow<2.6,>=2.5.0->tensorflow_text) (0.2.8)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<2,>=1.6.3->tensorboard~=2.5->tensorflow<2.6,>=2.5.0->tensorflow_text) (4.2.2)
    Requirement already satisfied: rsa<5,>=3.1.4; python_version >= "3.6" in /usr/local/lib/python3.7/dist-packages (from google-auth<2,>=1.6.3->tensorboard~=2.5->tensorflow<2.6,>=2.5.0->tensorflow_text) (4.7.2)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard~=2.5->tensorflow<2.6,>=2.5.0->tensorflow_text) (4.0.1)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.5->tensorflow<2.6,>=2.5.0->tensorflow_text) (1.3.0)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth<2,>=1.6.3->tensorboard~=2.5->tensorflow<2.6,>=2.5.0->tensorflow_text) (0.4.8)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata; python_version < "3.8"->markdown>=2.6.8->tensorboard~=2.5->tensorflow<2.6,>=2.5.0->tensorflow_text) (3.4.1)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.5->tensorflow<2.6,>=2.5.0->tensorflow_text) (3.1.0)
    Installing collected packages: tensorflow-text
    Successfully installed tensorflow-text-2.5.0
    


```python
#Importing Liabries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import time
```


```python
#Loading Data into DataFrame
data = pd.read_csv('data/preprocessed_reviews.csv')
data = data[:15000] # we are limiting ourself to 15k Reviews because of Computational Constraint
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Text</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>i recently purchased a bottle each of the suga...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>this is an excellent gluten free alternative t...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>my dog and i both love this product my vet eve...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>your friends and family may envy you when you ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>these should have an age limit i do not think ...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
def removeDup(df):
  if df.duplicated().any() == True:
    print("Found {} duplicated".format(df.duplicated().sum()))
    df.drop_duplicates(inplace=True, ignore_index=True)
    print("Duplicated removed")
    return df
  else:
    print("Duplicates not found")
    return df

data = removeDup(data)
```

    Found 34 duplicated
    Duplicated removed
    


```python
#Checking and removing Null values if any

def removeNull(df):
  "This function will check Duplicated and remove if any "
  if df.isnull().any().any() == True:
    print("Found {} Null values".format(df.isnull().values.sum()))
    df.dropna(inplace=True)
    print("Null Values Dropped")
    return df
  else:
    print("No Null value found")
    return df

data = removeNull(data)
```

    Found 2 Null values
    Null Values Dropped
    


```python
#Spliting tha data into train test

train_size = 0.668  #Spliting to get nearly 10k in training and 5k in testing
train_data, test_data = train_test_split(data, train_size= train_size, shuffle=True)
print("Length of Train Data:",len(train_data))
print("Length of Test Data:",len(test_data))
```

    Length of Train Data: 9995
    Length of Test Data: 4969
    

**MLP Model - bert Large as embedding layer**


```python
#Importing Bert pretrained model from Tensorflow Hub
preprocess_model_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
bert_encoder_model_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4'

precessing_layer = hub.KerasLayer( preprocess_model_url)
bert_layer = hub.KerasLayer( bert_encoder_model_url, trainable = False )
```


```python
#Creating MLP Model

def normalModel(precessing_layer, bert_layer):
  input = tf.keras.layers.Input(shape=(), dtype=tf.string)
  encorder_input = precessing_layer(input)
  encorder_output = bert_layer(encorder_input)
  final_output = encorder_output['pooled_output']
  x = tf.keras.layers.Dense(1024, activation='relu')(final_output)
  x = tf.keras.layers.Dense(2048, activation='relu')(x)
  x = tf.keras.layers.Dense(2048, activation='relu')(x)
  x = tf.keras.layers.Dense(1024, activation='relu')(x)
  x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
  model = tf.keras.Model(input, x)

  model.compile(loss='BinaryCrossentropy' , optimizer = Adam(learning_rate=1e-5), metrics= 'BinaryAccuracy' )
  return model
```

**N.P - Bert large model is only used for converting our text to numerical form so that we can apply MLP here Which makes it a MLP model**


```python
mlp_model = normalModel(precessing_layer, bert_layer)
mlp_model.summary()
```

    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None,)]            0                                            
    __________________________________________________________________________________________________
    keras_layer (KerasLayer)        {'input_word_ids': ( 0           input_1[0][0]                    
    __________________________________________________________________________________________________
    keras_layer_1 (KerasLayer)      {'encoder_outputs':  335141889   keras_layer[0][0]                
                                                                     keras_layer[0][1]                
                                                                     keras_layer[0][2]                
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 1024)         1049600     keras_layer_1[0][25]             
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 2048)         2099200     dense[0][0]                      
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 2048)         4196352     dense_1[0][0]                    
    __________________________________________________________________________________________________
    dense_3 (Dense)                 (None, 1024)         2098176     dense_2[0][0]                    
    __________________________________________________________________________________________________
    dense_4 (Dense)                 (None, 1)            1025        dense_3[0][0]                    
    ==================================================================================================
    Total params: 344,586,242
    Trainable params: 9,444,353
    Non-trainable params: 335,141,889
    __________________________________________________________________________________________________
    


```python
%%time

#Training our MLP Model
mlp_model_history = mlp_model.fit(x = train_data['Text'].to_numpy(), y = train_data['Score'].to_numpy(),batch_size = 32,
                    validation_data = (test_data['Text'].to_numpy(), test_data['Score'].to_numpy() ),
                               epochs=5)
```

    Epoch 1/5
    313/313 [==============================] - 267s 805ms/step - loss: 0.6787 - binary_accuracy: 0.5681 - val_loss: 0.6807 - val_binary_accuracy: 0.5506
    Epoch 2/5
    313/313 [==============================] - 251s 803ms/step - loss: 0.6375 - binary_accuracy: 0.6498 - val_loss: 0.6071 - val_binary_accuracy: 0.6981
    Epoch 3/5
    313/313 [==============================] - 251s 803ms/step - loss: 0.5897 - binary_accuracy: 0.6988 - val_loss: 0.5485 - val_binary_accuracy: 0.7519
    Epoch 4/5
    313/313 [==============================] - 251s 803ms/step - loss: 0.5392 - binary_accuracy: 0.7381 - val_loss: 0.5015 - val_binary_accuracy: 0.7744
    Epoch 5/5
    313/313 [==============================] - 251s 803ms/step - loss: 0.4962 - binary_accuracy: 0.7677 - val_loss: 0.4742 - val_binary_accuracy: 0.7802
    CPU times: user 6min 22s, sys: 3min 3s, total: 9min 25s
    Wall time: 21min 35s
    

**MLP model - Bert Small as Embedding layer**


```python
preprocess_model_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
bert_small_model_url = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'

precessing_layer = hub.KerasLayer( preprocess_model_url)
bert_small_layer = hub.KerasLayer( bert_small_model_url, trainable = False )
```


```python
#Creating MLP model 

def small_model(precessing_layer, bert_layer):
  input = tf.keras.layers.Input(shape=(), dtype=tf.string)
  encorder_input = precessing_layer(input)
  encorder_output = bert_layer(encorder_input)
  final_output = encorder_output['pooled_output']
  x = tf.keras.layers.Dense(512, activation='relu')(final_output)
  x = tf.keras.layers.Dense(1024, activation='relu')(x)
  x = tf.keras.layers.Dense(1024, activation='relu')(x)
  x = tf.keras.layers.Dense(512, activation='relu')(x)
  x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
  model = tf.keras.Model(input, x)
  
  model.compile(loss='BinaryCrossentropy' , optimizer = Adam(learning_rate=1e-5), metrics= 'BinaryAccuracy' )
  return model
```

**N.P - Bert small model is only used for converting our text to numerical form so that we can apply MLP here Which makes it a MLP model**


```python
bert_small_model = small_model(precessing_layer, bert_small_layer)
bert_small_model.summary()
```

    Model: "model_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_2 (InputLayer)            [(None,)]            0                                            
    __________________________________________________________________________________________________
    keras_layer_2 (KerasLayer)      {'input_word_ids': ( 0           input_2[0][0]                    
    __________________________________________________________________________________________________
    keras_layer_3 (KerasLayer)      {'sequence_output':  28763649    keras_layer_2[0][0]              
                                                                     keras_layer_2[0][1]              
                                                                     keras_layer_2[0][2]              
    __________________________________________________________________________________________________
    dense_5 (Dense)                 (None, 512)          262656      keras_layer_3[0][5]              
    __________________________________________________________________________________________________
    dense_6 (Dense)                 (None, 1024)         525312      dense_5[0][0]                    
    __________________________________________________________________________________________________
    dense_7 (Dense)                 (None, 1024)         1049600     dense_6[0][0]                    
    __________________________________________________________________________________________________
    dense_8 (Dense)                 (None, 512)          524800      dense_7[0][0]                    
    __________________________________________________________________________________________________
    dense_9 (Dense)                 (None, 1)            513         dense_8[0][0]                    
    ==================================================================================================
    Total params: 31,126,530
    Trainable params: 2,362,881
    Non-trainable params: 28,763,649
    __________________________________________________________________________________________________
    


```python
%%time

#Training our MLP model
bert_small_history = bert_small_model.fit(x = train_data['Text'].to_numpy(), y = train_data['Score'].to_numpy(),batch_size = 32,
                    validation_data = (test_data['Text'].to_numpy(), test_data['Score'].to_numpy() ),
                               epochs=5)
```

    Epoch 1/5
    313/313 [==============================] - 40s 120ms/step - loss: 0.5747 - binary_accuracy: 0.7235 - val_loss: 0.4948 - val_binary_accuracy: 0.7627
    Epoch 2/5
    313/313 [==============================] - 37s 119ms/step - loss: 0.4678 - binary_accuracy: 0.7774 - val_loss: 0.4548 - val_binary_accuracy: 0.7909
    Epoch 3/5
    313/313 [==============================] - 37s 119ms/step - loss: 0.4449 - binary_accuracy: 0.7919 - val_loss: 0.4477 - val_binary_accuracy: 0.7977
    Epoch 4/5
    313/313 [==============================] - 37s 119ms/step - loss: 0.4335 - binary_accuracy: 0.7946 - val_loss: 0.4470 - val_binary_accuracy: 0.7961
    Epoch 5/5
    313/313 [==============================] - 37s 118ms/step - loss: 0.4254 - binary_accuracy: 0.7983 - val_loss: 0.4346 - val_binary_accuracy: 0.8014
    CPU times: user 2min 57s, sys: 36.4 s, total: 3min 33s
    Wall time: 3min 8s
    

**Bert Large Trainable model**


```python
#Importing Bert pretrained model from Tensorflow Hub
preprocess_model_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
bert_encoder_model_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4'

precessing_layer = hub.KerasLayer( preprocess_model_url)
bert_layer = hub.KerasLayer( bert_encoder_model_url, trainable = True )
```


```python
#Creating Bert model

def bert_Model(precessing_layer, bert_layer):
  input = tf.keras.layers.Input(shape=(), dtype=tf.string)
  encorder_input = precessing_layer(input)
  encorder_output = bert_layer(encorder_input)
  final_output = encorder_output['pooled_output']
  x = tf.keras.layers.Dense(1, activation='sigmoid')(final_output)
  model = tf.keras.Model(input, x)

  model.compile(loss='BinaryCrossentropy' , optimizer = Adam(learning_rate=1e-5), metrics= 'BinaryAccuracy' )
  return model
```


```python
bert_model = bert_Model(precessing_layer, bert_layer)
bert_model.summary()
```

    Model: "model_2"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_3 (InputLayer)            [(None,)]            0                                            
    __________________________________________________________________________________________________
    keras_layer_4 (KerasLayer)      {'input_word_ids': ( 0           input_3[0][0]                    
    __________________________________________________________________________________________________
    keras_layer_5 (KerasLayer)      {'encoder_outputs':  335141889   keras_layer_4[0][0]              
                                                                     keras_layer_4[0][1]              
                                                                     keras_layer_4[0][2]              
    __________________________________________________________________________________________________
    dense_10 (Dense)                (None, 1)            1025        keras_layer_5[0][25]             
    ==================================================================================================
    Total params: 335,142,914
    Trainable params: 335,142,913
    Non-trainable params: 1
    __________________________________________________________________________________________________
    


```python
%%time

#Training our Bert Large Model
bert_model_history = bert_model.fit(x = train_data['Text'].to_numpy(), y = train_data['Score'].to_numpy(),batch_size = 16,
                    validation_data = (test_data['Text'].to_numpy(), test_data['Score'].to_numpy() ),
                               epochs=3)

```

    Epoch 1/3
    625/625 [==============================] - 632s 977ms/step - loss: 0.2468 - binary_accuracy: 0.8961 - val_loss: 0.1819 - val_binary_accuracy: 0.9288
    Epoch 2/3
    625/625 [==============================] - 609s 975ms/step - loss: 0.1164 - binary_accuracy: 0.9587 - val_loss: 0.1819 - val_binary_accuracy: 0.9278
    Epoch 3/3
    625/625 [==============================] - 609s 975ms/step - loss: 0.0639 - binary_accuracy: 0.9784 - val_loss: 0.2033 - val_binary_accuracy: 0.9362
    CPU times: user 18min 14s, sys: 8min 1s, total: 26min 15s
    Wall time: 30min 51s
    
