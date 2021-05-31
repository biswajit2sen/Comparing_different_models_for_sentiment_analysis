# Comparing Different Models for Sentiment analysis.

In this project I am trying to compare the performance and training time of different models for Sentiment analysis of reviews given by users for different products in amazon. <br />
I have used two LSTM models and one BERT large model which we will train on our data going further.<br />
For preprocessing of Data refer notebook : [Data_preprocessing.ipynb](https://github.com/biswajit2sen/Sentiment_analysis_amazon_fine_food_review/blob/main/Data_preprocessing.ipynb). <br />
For Comparing Different Models refer notebook: [Comparing_different_models_for_sentiment_analysis.ipynb](https://github.com/biswajit2sen/Comparing_different_models_for_sentiment_analysis/blob/main/Comparing_different_models_for_sentiment_analysis..ipynb)

## Data Collection
we get our data from : https://www.kaggle.com/snap/amazon-fine-food-reviews <br />
This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. <br />

## Data Preprocessing 

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
with open('data/Reviews.csv') as f:
  reader = csv.reader(f)
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
```


```python
pd.set_option('display.max_colwidth',None) #to disp max width of a columns
data.sample(5)
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
      <th>266571</th>
      <td>This popcorn is delicious, low- fat, low-calorie and gluten free. Most importantly its non GMO, which is difficult to find in popcorn.&lt;br /&gt;&lt;br /&gt;This service is superb! Speedy delivery, excellent service!!</td>
      <td>5</td>
    </tr>
    <tr>
      <th>336617</th>
      <td>I love this espresso coffee, it makes such a nice cup of espresso with an amazing crema.  It's very consistent and the flavor is smooth, however I have noticed that I get a major headache after drinking an espresso made with this coffee.  I am making it properly with the right amount of coffee so I know it's not that.  It's also not the caffeine since I drink drip coffee almost every day.  That leaves really one thing which is the robusta beans that is blended with arabica to make this coffee.&lt;br /&gt;&lt;br /&gt;Great coffee, just not for me and if anyone else experiences this then it's the robusta beans so just try another 100% arabic blend, it comes out good anyways.</td>
      <td>5</td>
    </tr>
    <tr>
      <th>290106</th>
      <td>Much better than pre-ground coffee in the can.  The best part was that it was still Folgers...which is good pre-ground or otherwise!</td>
      <td>5</td>
    </tr>
    <tr>
      <th>182733</th>
      <td>This assortment of cheeses is very mild, and extremely delicious and mouth watering too...The only difference is the age of the cheeses..."The quality of these cheeses is superb".I believe you will love them all..Great for the holidays,or year round..and highly recommended 5 star rated...enjoy !!! stewart l  12/03/05.</td>
      <td>5</td>
    </tr>
    <tr>
      <th>226668</th>
      <td>This was the fourth brand I tried due to my female shorthair cat's frequent vomiting without any sign of hairballs.  She started out with original Cat Chow, then I tried Eukanuba hairball formula, and eventually Science Diet sensitive stomach formula, which was by far the most problematic, causing her to throw up at least once daily.  My vet recommended a natural brand, and a Petsmart employee suggested the Blue Buffalo indoor formula specifically.  That was about three weeks ago, and knock on wood, she hasn't vomited once since.  Not only that, but she's actually become more affectionate now that she's not in a constant state of discomfort and stomach distress.  Her chronic dandruff has also almost completely disappeared, and both of my cats are shedding quite a bit less.  Their coats are softer and shinier than they've ever been before, and to top it all off, since Blue Buffalo has no fillers, they're actually getting full while eating less of it, and are defecating less frequently.  This food was truly a godsend for my sensitive-stomach kitty, and the additional coat benefits were amazing and completely unexpected.  I will offer the warning, however, that although my cats took to BB immediately when it was slowly mixed in with their old food, my mom's cats refuse to eat it.  I recommend buying different flavors one at a time in the smallest quantity available until you make sure you've found one your cat(s) will eat.  Definitely worth the price, especially if, like me, you've been buying expensive Eukanuba or Science Diet already!</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



**Here we can see that The Reviews Text have some Noice in them so we will Preprocess the text in a while**


```python
pd.reset_option("max_colwidth") #to reset
```


```python
#We have '5' scores for the reviews
#We coinsider '1' & '2' as Negative Reviews
#we coinsider '4' & '5' as Positive Reviews
data['Score'] = data['Score'].map({1:0, 2:0, 4:1, 5:1, 3:3})
#We are remove Reviews with Score '3' bcz they are neighter Positive or negative,
#they are coinsidered to be Neutral
data = data[data['Score']!=3]
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.
    


```python
print("total no of reviews after removing neutral reviews: {}".format(len(data)))
```

    total no of reviews after removing neutral reviews: 525814
    


```python
#Checking and Removing Deplicates if any

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

    Found 161973 duplicated
    Duplicated removed
    


```python
#Checking and removing Null values if any

def removeNull(df):
  """ This function will check Null and remove if any """
  
  if df.isnull().any().any() == True:
    print("Found {} Null Values".format(df.isnull().sum()))
    df.dropna(inplace=True)
    print("Null Values Dropped")
    return df
  else:
    print("No Null value found")
    return df

data = removeNull(data)
```

    No Null value found
    


```python
#converting our Text column to str and Score column to int type
def colConvert(df, column_name, convtotype):
  if (df[column_name].map(type)!=convtotype).any() == True:
    print("Different Row type found")
    df[column_name] = df[column_name].astype(convtotype)
    print("All Rows are Converted to {}".format(convtotype) )
    return df
  else:
    print("All Rows of type {}".format(convtotype))
    return df 

data = colConvert(data, 'Text', str)
data = colConvert(data, 'Score', int)
```

    All Rows of type <class 'str'>
    All Rows of type <class 'int'>
    


```python
print("Total no of reviews now: {}".format(len(data)))
```

    Total no of reviews now: 363841
    


```python
#no of Posive reviews
print("Positive Reviews:",len(data[data['Score']==1]))
```

    Positive Reviews: 306768
    


```python
#no of negative reviews
neg_rev = len(data[data['Score']==0])
print("Negative Reviews:",neg_rev)
```

    Negative Reviews: 57073
    

We can see its a highly imbalance data so we will try to balance it by removing excess positive reviews


```python
# creating Negative Reviews Df 
negative_review_df = data[data['Score']==0]
```


```python
# keeping Positive Reviews Df 
data = data[data['Score']==1]
# keeping same number of Positive Reviews as of negative
data = data[:neg_rev]
```


```python
#Adding back to the dataframe
data = data.append([negative_review_df], ignore_index=True)
```


```python
# shuffle the DataFrame rows
data = data.sample(frac = 1)
```

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

    100%|██████████| 114146/114146 [00:51<00:00, 2214.22it/s]
    


```python
#Checking for the review text again
pd.set_option('display.max_colwidth',None) #to disp max width of columns
preprocessed_reviews.sample(5) # To randomly check 5 Rows
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
      <th>11864</th>
      <td>this marmalade is so light and tasty on toast sellers are wonderful and pack beautifully also recommend the prickly pear jelly as chicken glaze and prickly pear syrup add a bit to homemade lemonade and enjoy</td>
      <td>1</td>
    </tr>
    <tr>
      <th>65110</th>
      <td>these noodles just arrived yesterday and i now know the secret of their weight loss properties they will have you throwing up all night i ate some of these for dinner last night and though i did not really like them i thought i could at least tolerate them but i started feeling very full after i would eaten a small ammount which i thought was a good thing at the time and ended up waking up at this morning with intense stomach pain nausea and vomiting i have never had stomach pain this intense in my life and i have a very strong stomach so this was a horrible shock i would never buy or reccomend this to anyone ever</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95685</th>
      <td>this stuff has an off smell i did not even bother using it i am going with wabash since the little sample of coconut oil i got with their popcorn maker actually smells extremely pleasant not this stuff and they are supposed to be the same coconut oil with a touch of coloring</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7920</th>
      <td>these are incredible eating these makes me not want to eat candy i love these the price is not that great though i think blue diamond should save money by packaging them in something cheaper than plastic i did notice it on amazon in a box but the price looks odd over too bad amazon does not stock this so no free shipping</td>
      <td>1</td>
    </tr>
    <tr>
      <th>58425</th>
      <td>if this was called medium roast i would give it stars it is a good coffee but this is called french vanilla caramel and this review is based on the product is name let me say that vanilla and caramel are perhaps my favorite flavors coffee ice cream candy you name it if it is vanilla or caramel i love it with the exception of this where is the vanilla where is the caramel and do not get me wrong i am not trying to say that the flavors are too mild i am saying that they are non existent and therefore based on the name this coffee gets star</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.reset_option("max_colwidth") #to reset
```

**Noice has been removed from the text**


```python
preprocessed_reviews.to_csv('data/preprocessed_reviews.csv', index=False )
```
## EDA
## Different Models Comparison



## Conclusion

## Technology used

## Credits
