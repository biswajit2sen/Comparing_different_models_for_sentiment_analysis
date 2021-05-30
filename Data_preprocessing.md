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
import os
os.environ['KAGGLE_CONFIG_DIR'] = "/content/gdrive/My Drive/ML-in-colab"
```


```python
#downloading kaggle dataset
!kaggle datasets download -d snap/amazon-fine-food-reviews -p data
```

    Downloading amazon-fine-food-reviews.zip to data
     97% 235M/242M [00:01<00:00, 140MB/s]
    100% 242M/242M [00:01<00:00, 140MB/s]
    


```python
#we can check the content by ls command
!ls data
```

    amazon-fine-food-reviews.zip
    


```python
#Unziping the file into data folder
!unzip data/\*.zip -d data
```

    Archive:  data/amazon-fine-food-reviews.zip
      inflating: data/Reviews.csv        
      inflating: data/database.sqlite    
      inflating: data/hashes.txt         
    


```python
#checking the file content
!ls data
```

    amazon-fine-food-reviews.zip  database.sqlite  hashes.txt  Reviews.csv
    


```python
#removing zip file from data folder
!rm data/*.zip
```


```python
!ls data
```

    database.sqlite  hashes.txt  Reviews.csv
    


```python
#importing required libraries
import pandas as pd
import re
from bs4 import BeautifulSoup
import csv
```


```python
#Reading the file
file = pd.read_csv('data/Reviews.csv')
```


```python
#Checking the number of total reviews
reviews = 0
with open('data/Reviews.csv') as file:
  reader = csv.reader(file)
  for row in reader:
    reviews +=1
print("Total number of Reviews is: {}".format(reviews))
```

    Total number of Reviews is: 568455
    


```python
file.head()
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
      <th>Id</th>
      <th>ProductId</th>
      <th>UserId</th>
      <th>ProfileName</th>
      <th>HelpfulnessNumerator</th>
      <th>HelpfulnessDenominator</th>
      <th>Score</th>
      <th>Time</th>
      <th>Summary</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>B001E4KFG0</td>
      <td>A3SGXH7AUHU8GW</td>
      <td>delmartian</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1303862400</td>
      <td>Good Quality Dog Food</td>
      <td>I have bought several of the Vitality canned d...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>B00813GRG4</td>
      <td>A1D87F6ZCVE5NK</td>
      <td>dll pa</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1346976000</td>
      <td>Not as Advertised</td>
      <td>Product arrived labeled as Jumbo Salted Peanut...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>B000LQOCH0</td>
      <td>ABXLMWJIXXAIN</td>
      <td>Natalia Corres "Natalia Corres"</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1219017600</td>
      <td>"Delight" says it all</td>
      <td>This is a confection that has been around a fe...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>B000UA0QIQ</td>
      <td>A395BORC6FGVXV</td>
      <td>Karl</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>1307923200</td>
      <td>Cough Medicine</td>
      <td>If you are looking for the secret ingredient i...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>B006K2ZZ7K</td>
      <td>A1UQRSCLF8GW1T</td>
      <td>Michael D. Bigham "M. Wassir"</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>1350777600</td>
      <td>Great taffy</td>
      <td>Great taffy at a great price.  There was a wid...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#keeping the Required columns in DF
data = file[['Text','Score']]
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
      <td>I have bought several of the Vitality canned d...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Product arrived labeled as Jumbo Salted Peanut...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>This is a confection that has been around a fe...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>If you are looking for the secret ingredient i...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Great taffy at a great price.  There was a wid...</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.set_option('display.max_rows', None)
```


```python
#We have '5' scores for the reviews
#We coinsider '1' & '2' as Negative Reviews
#we coinsider '4' & '5' as Positive Reviews
data['Score'] = data['Score'].map({1:0, 2:0, 4:1, 5:1})
#We are remove Reviews with Score '3' bcz they are neighter Positive or negative,
#they are coinsidered to be Neutral
data = data[data['Score']!=3]
```


```python
#tdf['score'] = tdf['Score'].map({1:0, 2:0, 4:1, 5:1})
#tdf.head(100)
print("total no of reviews now: {}".format(len(data)))
```

    total no of reviews now: 525814
    


```python
#converting float to int
data['Score'] = data['Score'].astype(int)
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
      <td>I have bought several of the Vitality canned d...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Product arrived labeled as Jumbo Salted Peanut...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>This is a confection that has been around a fe...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>If you are looking for the secret ingredient i...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Great taffy at a great price.  There was a wid...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Text Preprocessing


```python
#Defining StopWords
stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])
```


```python
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def text_preprocess(df, text_column_to_preprocess):

  from tqdm import tqdm
  preprocessed_reviews = []

  for sentance in tqdm(df[text_column_to_preprocess].values):
      sentance = re.sub(r"http\S+", "", sentance)
      sentance = BeautifulSoup(sentance, 'lxml').get_text()
      sentance = decontracted(sentance)
      sentance = re.sub("\S*\d\S*", "", sentance).strip()
      sentance = re.sub('[^A-Za-z]+', ' ', sentance)
      # https://gist.github.com/sebleier/554280
      # now using stopwords as of now
      sentance = ' '.join(e.lower() for e in sentance.split()) # if e.lower() not in stopwords)
      preprocessed_reviews.append(sentance.strip())
  
  df[text_column_to_preprocess] = preprocessed_reviews
  return df
```


```python
preprocessed_reviews = text_preprocess(data, 'Text')
```

    100%|██████████| 525814/525814 [03:56<00:00, 2219.08it/s]
    


```python
preprocessed_reviews.sample()
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
      <th>182195</th>
      <td>i have been using this product for more that y...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#no of Posive recviews
print("Positive Reviews:",len(preprocessed_reviews[preprocessed_reviews['Score']==1]))
```

    Positive Reviews: 443777
    


```python
#no of negative recviews
print("Negative Reviews:",len(preprocessed_reviews[preprocessed_reviews['Score']==0]))
```

    Negative Reviews: 82037
    

We can see its a highly imbalance data so we will add more negative data by duplicating the existing negative reviews


```python
# creating Negative Reviews Df for adding
negative_review_df = preprocessed_reviews[preprocessed_reviews['Score']==0]
```


```python
#Adding back to the dataframe
preprocessed_reviews = preprocessed_reviews.append([negative_review_df,negative_review_df,negative_review_df,negative_review_df], ignore_index=True)
```


```python
# shuffle the DataFrame rows
preprocessed_reviews = preprocessed_reviews.sample(frac = 1)
```


```python
preprocessed_reviews.to_csv('data/preprocessed_reviews.csv', index=False )
```
