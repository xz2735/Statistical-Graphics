Machine Learning in Facebook
===
© 2018 AlgoLib Quant team

# 1. Machine learning in social media analysis
## 1.1 What is Machine Learning?
Arthur Samuel, an early pioneer in the field of artificial intelligence, defined machine learning as “the subfield of computer science that gives computers the ability to learn without being explicitly programmed.” Additionally, Wikipedia describes machine learning as “a scientific discipline that explores the construction and study of algorithms that can learn from data.” Machine learning tasks can be classified into two categories, such as Supervised learning and Unsupervised learning based on the existence of feedback and signal.
![](https://i.imgur.com/vet4r36.png)

* **Supervised Learning**

In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.
![](https://i.imgur.com/5gGxD0A.png)

Supervised learning problems are categorized into "**regression**" and "**classification**" problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some **continuous** function. In a classification problem, we are instead trying to predict results in a **discrete** output. In other words, we are trying to map input variables into discrete categories. 
* **Unsupervised Learning**


Unsupervised learning, on the other hand, allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by clustering the data based on relationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results, i.e., there is no teacher to correct you.

## 1.2 What is Social Media Analysis?
“Social media isn’t just about collecting Facebook “likes” anymore”, says Danny Bradbury, and we couldn’t agree more. Nowadays, social media is not just a place for people to connect with each other. It is also a perfect platform for companies to find  customers‘ hidden insights based on social conversations, posts containing words, pictures and videos, usually called unstructured data, to enable informed and insightful decision making. Social media analysis can help to answer the following questions:
* What drives people to purchase my product?
* How are customers using my product?
* What consumer trends provide the biggest business opportunities?

## 1.3 Why Machine Learning for social media analysis?
Human is good at learning and understanding language. However, with trillions of social media posts to be analysed, it is unrealistic to rely on manual social media analysis.  Here, machine learning algorithms combined with human understanding with high speed can be trained to categorize posts just as what human can do. For example, when people said 'I love apple', does 'apple' mean the fruit or the electronic devices? 
![](https://i.imgur.com/bAqptmP.jpg)
![](https://i.imgur.com/AqHmNam.png)

Of course, machine learning does not end on this. There are basically three applications for machine learning technology in social analysis, including improvements in **sentiment analysis**, **audience analysis** and **image analysis** and we will refer them later.

# 2. About Facebook
## 2.1 What is Facebook?
Social media platforms – such as Facebook, Twitter, Instagram, and Tumblr – are a public ‘worldwide forum for expression’, where billions of people connect and share their experiences, personal views, and opinions about everything from vacations to live events.

Facebook is the most popular online social networking media in the world, consisting of a series of interrelated profile pages in which members post a broad range of information about themselves and link their own profile to others profiles.
## 2.2 Platforms for Machine Learning in Facebook
### 2.2.1 Natural Language Process(NLP)
#### 2.2.1.1 Historical Solution
In traditional NLP methods, words are converted into a format that computer can recognize. For example, the word “brother” might be assigned an integer ID such as 4598, while the word “bro” becomes another integer, like 986665. This representation requires each word to be seen with exact spellings in the training data to be understood.
#### 2.2.1.2 Deep Learning Solution
Now, we can use [word embeddings](https://en.wikipedia.org/wiki/Word_embedding), a mathematical concept that create the semantic relationship among words. Thus, we can not only capture the deeper semantic meaning of words, but also understand the same semantics across multiple languages, despite differences in the surface form. Word embeddings, is proved to be currently optimized solution for NLP tasks, such as sentiment analysis. Facebook bulit **DeepText**, a deep learning-based text understanding engine that is able to understand contents of text in several thousands of posts per seconds with about 20 languages. 

![](https://i.imgur.com/udTvxez.png)

In DeepText, **FbLearner**, a platform developed by Facebook AI Research Team to reuse algorithms in different products, scaling to run thousands of simultaneous custom experiments, and managing experiments with ease, is used for model training.

# 3. Detailed Applications for Social Media Analysis with ML
## 3.1 Sentiment Analysis
### 3.1.1 What is Sentiment Analysis?
In January, 2017, mentions about Toyota rocketed unusually in social media, it seemed the spike of talks proved the increasing popularity of the brand. However, through deep investigations, people found most of the conversations about Toyota were actually negative. What happened to Toyota? People finally figured out it stemmed from a tweet of a celebrity, Donald Trump. Without sentiment analysis, Toyota may miss this important information and failed to take actions in time.
![](https://i.imgur.com/BlIicKa.png)
![](https://i.imgur.com/gUBy6ZN.png)

In general, sentiment analysis is the measurement of positive language and negative language. It is always significant for companies and brands to know how people feel about their products or services, so that company could quickly make decisions about what to do next.
By sentiment analysis, companies can identify a potential crisis, like what we mention above about Toyota, help to evaluate overall brand health, track the reaction to new products or branding, compare yourself to your competition and market analysis.

Sentiment analysis can also go beyond positive and negative. There is also a more interesting example to help understand it. When someone focused on how people feel about Apple Watch from Twitter posts through sentiment analysis, the following word clouds obtained relating to the Apple watch highlight some key points which give a strong insight into the public reaction to the Apple Watch.



![](https://i.imgur.com/VzEx4ph.png)
Positive:

People “want” one!
They think it “looks” “pretty”
They like that it comes in “34 styles”
They are “excited” by it.

Negative:

Some think it looks “ugly”
They are wary of the “battery life”
In saying that it’s also pretty interesting that both the positive and negative tweets tell us that most people just care about how it looks over any of its fancy features!

### 3.1.2 Detailed solution and codes for sentiment analysis
There is a breaking news in June, 2017. It is reported that Lee Wei Ling, Lee Hsien Yang issue statement to say they have 'lost confidence' in PM Lee Hsien Loong, their brother.

[![](https://i.imgur.com/KXC30CT.png)](http://www.asiaone.com/singapore/lee-wei-ling-lee-hsien-yang-issue-statement-say-they-have-lost-confidence-pm-lee-hsien)

So we wonder what happened to them and what people thought about this event. Now, we try to collect public response to these statements based on Facebook comments. 

**Get Started**

Firstly, to deal with this, we need to install Python3, then we should go to the **Facebook Graph API** at https://developers.facebook.com/tools/explorer/ to download the relevant posts in Facebook. 
![](https://i.imgur.com/r8H65A1.png)
Then we will use the requests and google-cloud-language libraries for making HTTP requests and performing sentiment analysis. To install these, run:

``` 
#if you are using the python3 command, you may need to use pip3 here as well
pip install --upgrade requests google-cloud-language
```
(Update 2018–01–05) This post has been updated to use the Google Cloud Python Library v0.26.1 and newer. If you encounter errors similar to AttributeError: module 'google.cloud.language' has no attribute 'LanguageServiceClient' then you could try rerunning the above command to update your packages.


Unlike AWS, resources on Google Cloud Platform are grouped by project. Go to https://cloud.google.com/natural-language/docs/getting-started and follow steps 1–6 to set up a project. You may also need to install the Google Cloud SDK to use gcloud.

**Downloading Facebook comments**

Our objective here is to download all the comments on LHL’s post on his Facebook page responding to his siblings:
https://www.facebook.com/leehsienloong/posts/1505690826160285
We will do this by traversing the **Facebook Graph API**.

**Simple introduction about Facebook Graph API**

The Graph API is the primary way for apps to read and write to the Facebook social graph. It exposes facebook data, comprised of connected entities, such as a Facebook User, a Page, and even a Comment.

Entities are linked by vertices, which are the properties of an entity. For example, a Post entity is linked to many comment entities represent each comment on that post.

We need to find the Post entity corresponds to LHL’s post, and go through all the Comment entities connected to it. To simplify things, the post entity is referenced by an ID comprised of to the ID of the user or page who made the post, and the ID of the post itself, which can be found in the post URL.


To get LHL’s page ID through the Graph API Explorer, we just enter his page username into the query box:
![](https://i.imgur.com/M1Mzer1.png)

After we get his page ID, we can find his post:
![](https://i.imgur.com/P0bDkCN.png)

And just by appending /comments to the post ID, we can get all the comments made on the post:
![](https://i.imgur.com/xOBSpdb.png)

However, we don’t want to use the Graph API Explorer to manually save all the comments, so we will use a Python script instead (updated 2018–01–15):
```python=
import requests
import signal
import sys

graph_api_version = 'v2.9'
access_token = 'YOUR_FACEBOOK_ACCESS_TOKEN_HERE'

# LHL's Facebook user id
user_id = '125845680811480'

# the id of LHL's response post at https://www.facebook.com/leehsienloong/posts/1505690826160285
post_id = '1505690826160285'

# the graph API endpoint for comments on LHL's post
url = 'https://graph.facebook.com/{}/{}_{}/comments'.format(graph_api_version, user_id, post_id)

comments = []

# set limit to 0 to try to download all comments
limit = 200


def write_comments_to_file(filename):
    print()

    if len(comments) == 0:
        print('No comments to write.')
        return

    with open(filename, 'w', encoding='utf-8') as f:
        for comment in comments:
            f.write(comment + '\n')

    print('Wrote {} comments to {}'.format(len(comments), filename))


# register a signal handler so that we can exit early
def signal_handler(signal, frame):
    print('KeyboardInterrupt')
    write_comments_to_file('comments.txt')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

r = requests.get(url, params={'access_token': access_token})
while True:
    data = r.json()

    # catch errors returned by the Graph API
    if 'error' in data:
        raise Exception(data['error']['message'])

    # append the text of each comment into the comments list
    for comment in data['data']:
        # remove line breaks in each comment
        text = comment['message'].replace('\n', ' ')
        comments.append(text)

    print('Got {} comments, total: {}'.format(len(data['data']), len(comments)))

    # check if we have enough comments
    if 0 < limit <= len(comments):
        break

    # check if there are more comments
    if 'paging' in data and 'next' in data['paging']:
        r = requests.get(data['paging']['next'])
    else:
        break

# save the comments to a file
write_comments_to_file('comments.txt')
``` 
By default, this script will just stop after downloading 200 comments. You can adjust the limit variable inside the script to download more, or set it to 0 to try to download everything.

This script works because we can get the same output from the Graph API Explorer by visiting the Graph API URL directly. This script does several things:
1. Make a HTTP request to get the comments on LHL’s post
2. Save the text of the comments on the post into a Python list
3. Check if there are any more comments (using the paging cursors returned in the request, refer to https://developers.facebook.com/docs/graph-api/using-graph-api for more information about paging).
4. Save the comments we got into a file.

If you want to know more about how to use Facebook Graph API by python, please look at the tutorial video at [this blog](https://www.kdnuggets.com/2017/06/6-interesting-things-facebook-python.html).

**Analysing the comment sentiment**

Now we have a comment list, that is ``` comments.txt```  file to analyse. We can use the Google Cloud Natural Language to get the sentiment of each comment, which could help to identify whether comments are positive, negative or neutral. The code is following:
```python=
import signal
import sys

from google.cloud import language, exceptions

# create a Google Cloud Natural Language API Python client
client = language.LanguageServiceClient()


# a function which takes a block of text and returns its sentiment and magnitude
def detect_sentiment(text):
    """Detects sentiment in the text."""

    document = language.types.Document(
        content=text,
        type=language.enums.Document.Type.PLAIN_TEXT)

    sentiment = client.analyze_sentiment(document).document_sentiment

    return sentiment.score, sentiment.magnitude


# keep track of count of total comments and comments with each sentiment
count = 0
positive_count = 0
neutral_count = 0
negative_count = 0


def print_summary():
    print()
    print('Total comments analysed: {}'.format(count))
    print('Positive : {} ({:.2%})'.format(positive_count, positive_count / count))
    print('Negative : {} ({:.2%})'.format(negative_count, negative_count / count))
    print('Neutral  : {} ({:.2%})'.format(neutral_count, neutral_count / count))


# register a signal handler so that we can exit early
def signal_handler(signal, frame):
    print('KeyboardInterrupt')
    print_summary()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

# read our comments.txt file
with open('comments.txt', encoding='utf-8') as f:
    for line in f:
        # use a try-except block since we occasionally get language not supported errors
        try:
            score, mag = detect_sentiment(line)
        except exceptions.BadRequest:
            # skip the comment if we get an error
            continue

        # increment the total count
        count += 1

        # depending on whether the sentiment is positive, negative or neutral, increment the corresponding count
        if score > 0:
            positive_count += 1
        elif score < 0:
            negative_count += 1
        else:
            neutral_count += 1

        # calculate the proportion of comments with each sentiment
        positive_proportion = positive_count / count
        neutral_proportion = neutral_count / count
        negative_proportion = negative_count / count

        print(
            'Count: {}, Positive: {:.3f}, Neutral: {:.3f}, Negative: {:.3f}'.format(
                count, positive_proportion, neutral_proportion, negative_proportion))

print_summary()
``` 
The output is like:
```
...
Count: 379, Positive: 0.657, Neutral: 0.190, Negative: 0.153
Count: 380, Positive: 0.655, Neutral: 0.192, Negative: 0.153
Count: 381, Positive: 0.656, Neutral: 0.192, Negative: 0.152
Count: 382, Positive: 0.657, Neutral: 0.191, Negative: 0.152
Count: 383, Positive: 0.658, Neutral: 0.191, Negative: 0.151
Count: 384, Positive: 0.659, Neutral: 0.190, Negative: 0.151
...
```
Finally, we get:
```
Total comments analysed: 781
Positive : 530 (67.86%)
Negative : 109 (13.96%)
Neutral  : 142 (18.18%)
```
**Conclusion**

It is obvious that nearly 70% of comments are positive. Thus this could be explained as a sign of strong support for the PM. However, it is also possible that visitors in Facebook are biased.

However, although this code could be quite trivial as the Google Cloud has done almost everything for you and you do not need to work on machine learning algorithm by yourself, it is a pity for people who desire to understand the procedure of machine learning algorithm. In the following, a much more complicated machine learning algorithm will be introduced to you.

### 3.1.3 Complicated machine learning algorithm and procedure
#### 3.1.3.1 Introduction 
 As we have talked about how to obtain data from API endpoint, so we now skip this step and assume we already have such data. The program is basically divided into 6 steps, that is
*  Import and parse the data sets.
*  Create feature columns to describe the data.
*  Select the type of model.
*  Train the model.
*  Evaluate the model's effectiveness.
*  Let the trained model make predictions.


Firstly, we import some libraries in python. Natural Language ToolKit (NLTK) is an incredible library for Python that can do a huge amount of text processing and analysis.
```python=
import re, math, collections, itertools
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
```
#### 3.1.3.2 Details of Each Part

**Import and parse the data sets**

Methods in import and load datasets are various, of course, you can download graph data from Facebook Graph API as mentioned above. 

Now import the dataset, these data can be found in https://github.com/abromberg/sentiment_analysis_python/tree/master/polarityData
```
POLARITY_DATA_DIR = os.path.join('polarityData', 'rt-polaritydata')
RT_POLARITY_POS_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-pos.txt')
RT_POLARITY_NEG_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-neg.txt')
```
**Describe the data**

A feature column is a data structure that tells your model how to interpret the data in each feature. Basically, features are characteristics of an entitle, while the label is the thing we're trying to predict. For example, given solely on the length and width of their sepals and petals, classify the different species of Iris flower. In this case, the flower species is the label that we are going to predict and other parameters are features.

In this block of codes, it helps to split up the reviews by line and then builds a posFeature variable containing the output of our feature selection mechanism with ‘pos’ or ‘neg’ appended to it, depending on whether the review it is drawing from is positive or negative.
```python=
#this function takes a feature selection mechanism and returns its performance in a variety of metrics
def evaluate_features(feature_select):
    posFeatures = []
	negFeatures = []
	#http://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
        #breaks up the sentences into lists of individual words (as selected by the input mechanism) and appends 'pos' or 'neg' after each list
	with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
		for i in posSentences:
			posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			posWords = [feature_select(posWords), 'pos']
			posFeatures.append(posWords)
	with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
		for i in negSentences:
			negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			negWords = [feature_select(negWords), 'neg']
			negFeatures.append(negWords)

```

Then, it separates the data into training and testing dataset. The training set contains the examples that we'll use to train the model; the test set contains the examples that we'll use to evaluate the trained model's effectiveness.

```python=
#selects 3/4 of the features to be used for training and 1/4 to be used for testing
    posCutoff = int(math.floor(len(posFeatures)*3/4))
    negCutoff = int(math.floor(len(negFeatures)*3/4))
    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]
```

**Select the type of model and train it**

A model represents a kind of relationship between features and the label. Usually, there are a large number of  interlacing mathematical functions and parameters in a complex machine learning models, so it is usually hard to summarize. 

In general, a good machine learning method can help to determine the relationship for you, if you feed enough representative examples into the right machine learning model type. Training is the stage of machine learning in which the model is gradually optimized (learned).

To specify a model type, instantiate an Estimator class. TensorFlow provides two categories of Estimators:

* pre-made Estimators, which someone else has already written for you.
* custom Estimators, which you must code yourself, at least partially.

In sentiment analysis, Naive Bayes Classifier, MaxEnt and SVM are often applied for classification. Now, train a pre-made Estimator called Naive Bayes Classifier with well separated dataset through NLTK.

```
classifier = NaiveBayesClassifier.train(trainFeatures)
```
**Evaluate the model's effectiveness**

Now, we need to check whether this model works well or not through the test dataset.

First, I have to initiate referenceSets and testSets, to be used shortly. ReferenceSets will contain the actual values for the testing data (which we know because the data is prelabeled) and testSets will contain the predicted output.

Next, for each one of the testFeatures (the reviews that need testing), I iterate through three things: an arbitrary ‘i’, so be used as an identifier, and then the features (or words) in the review, and the actual label (‘pos’ or ‘neg’).

I add the ‘i’ (the unique identifier) to the correct bin in referenceSets. I then predict the label based on the features using the trained classifier and put the unique identifier in the predicted bin in testSets.
```python=
#puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)
```
After that, it returns a big list of identifiers in referenceSets[‘pos’], which are the reviews known to be positive (and the same for the negative reviews). It also gives me a list of identifiers in testSets[‘pos’], which are the reviews predicted to be positive (and similarly for predicted negatives). Next, compare them and find how accurate this model is.

Then, we want to print out the comparison results.
```python=
    print('train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
    print('accuracy:', nltk.classify.util.accuracy(classifier, testFeatures))
    print('pos precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos']))
    print('pos recall:', nltk.metrics.recall(referenceSets['pos'], testSets['pos']))
    print('neg precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg']))
    print('neg recall:', nltk.metrics.recall(referenceSets['neg'], testSets['neg'])
    classifier.show_most_informative_features(10))
```

Next, we create a feature selection mechanism that uses all words and it is quite simple.
```	python=
def make_full_dict(words):
    return dict([(word, True) for word in words])
#tries using all words as the feature selection mechanism
print('using all words as features'
evaluate_features(make_full_dict))
```
The output is shown as following:

```
using all words as features
train on 7998 instances, test on 2666 instances
accuracy: 0.773068267067
pos precision: 0.787066246057
pos recall: 0.748687171793
neg precision: 0.760371959943
neg recall: 0.797449362341
Most Informative Features
              engrossing = True              pos : neg    =     17.0 : 1.0
                   quiet = True              pos : neg    =     15.7 : 1.0
                mediocre = True              neg : pos    =     13.7 : 1.0
               absorbing = True              pos : neg    =     13.0 : 1.0
                portrait = True              pos : neg    =     12.4 : 1.0
              refreshing = True              pos : neg    =     12.3 : 1.0
                   flaws = True              pos : neg    =     12.3 : 1.0
               inventive = True              pos : neg    =     12.3 : 1.0
                 triumph = True              pos : neg    =     11.7 : 1.0
            refreshingly = True              pos : neg    =     11.7 : 1.0
```

The next step is to select features and it is to only take the n most important features.

Firstly, split words and make them iterable.
```python=
def create_word_scores():
    #splits sentences into lines
    posSentences = open('polarityData\\rt-polaritydata\\rt-polarity-pos.txt', 'r')
    negSentences = open('polarityData\\rt-polaritydata\\rt-polarity-neg.txt', 'r')
    posSentences = re.split(r'\n', posSentences.read())
    negSentences = re.split(r'\n', negSentences.read())
 
    #creates lists of all positive and negative words
    posWords = []
    negWords = []
    for i in posSentences:
        posWord = re.findall(r"[\w']+|[.,!?;]", i)
        posWords.append(posWord)
    for i in negSentences:
        negWord = re.findall(r"[\w']+|[.,!?;]", i)
        negWords.append(negWord)
    posWords = list(itertools.chain(*posWords))
    negWords = list()
```

Here, we define the frequency distribution for counting positive and negative reviews, in order to obtain total count of those words.
```python=
word_fd = FreqDist()
cond_word_fd = ConditionalFreqDist()
for word in posWords:
        word_fd.inc(word.lower())
        cond_word_fd['pos'].inc(word.lower())
    for word in negWords:
        word_fd.inc(word.lower())
        cond_word_fd['neg'].inc(word.lower())
        pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count
```
Finally, through chi_squared test, We find each word’s positive information score and negative information score and add them up. Then select the best words.
```python=
word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
 
    return word_scores
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words
def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])
numbers_to_test = [10, 100, 1000, 10000, 15000]
#tries the best_word_features mechanism with each of the numbers_to_test of features
for num in numbers_to_test:
    print 'evaluating best %d word features' % (num)
    best_words = find_best_words(word_scores, num)
    evaluate_features(best_word_features)
```
Here is the output:
```
evaluating best 10 word features
train on 7998 instances, test on 2666 instances
accuracy: 0.574643660915
pos precision: 0.549379652605
pos recall: 0.830457614404
neg precision: 0.652841781874
neg recall: 0.318829707427    

evaluating best 100 word features
train on 7998 instances, test on 2666 instances
accuracy: 0.682295573893
pos precision: 0.659868421053
pos recall: 0.752438109527
neg precision: 0.712041884817
neg recall: 0.61215303826    

evaluating best 1000 word features
train on 7998 instances, test on 2666 instances
accuracy: 0.79632408102
pos precision: 0.817014446228
pos recall: 0.763690922731
neg precision: 0.778169014085
neg recall: 0.82895723931    

evaluating best 10000 word features
train on 7998 instances, test on 2666 instances
accuracy: 0.846586646662
pos precision: 0.868421052632
pos recall: 0.81695423856
neg precision: 0.827195467422
neg recall: 0.876219054764    

evaluating best 15000 word features
train on 7998 instances, test on 2666 instances
accuracy: 0.846961740435
pos precision: 0.862745098039
pos recall: 0.825206301575
neg precision: 0.832494608196
neg recall: 0.868717179295
```
For the full code, check out the [GitHub repository](https://github.com/abromberg/sentiment_analysis_python).
#### 3.1.3.4 Conclusion
Overall, the result is satisfying, as classifying reviews with over 80% accuracy is pretty impressive and we can see lots of applications for this technology. 

But of course, we still have some parts that can be improved. For example, we can try a different classifier different from Naive Bayes Classifier. And perhaps we can add different feature selection mechanisms.

## 3.2 Audience Analysis
### 3.2.1 What is audience analysis?

Audience analysis refers to researching the demographic, location, and other aspects of a group. Audience analysis can focus on audience for one brand or the same kind of product. It is usually very helpful to give a deep understanding in  the intricacies of consumer preferences, demographics and motivations of your audience, which helps brands or companies to make decision for updating their products or improving their advertising. Thus make products or brands reach the full potential. Today, social media data helps brands gain actionable business insights about an audience quickly.

For example, Arby’s Restaurant Group used data science to determine how visitors will behave if the restaurant is closed for remodeling. Their analysis has shown that remodeling will increase the number of visitors even taking into account that financial losses are unavoided because of the dead time. Due to this analysis, they have performed five remodels within a year to increase their annual income.

### 3.2.2 Use cases 

Actually, audience analysis can take many forms. Both sentiment analysis and image analysis are the ways to understand the targeted audience.

* **Locate an audience**
* **Understand key demographics**
![](https://i.imgur.com/U3ziAgk.png)
* **Track affinities and interests**
![](https://i.imgur.com/MpUFmy0.png)
* **Find new audiences**
* **Segment your audience**
![](https://i.imgur.com/ucRltYy.png)
* **Identify influencers**



### 3.2.3 Audience Analysis in Facebook 
**Facebook Audience Prediction & Optimization Technology**

As a Facebook Marketing Partner with patented processes in natural language processing and semantic analysis, **CitizenNet** creates transparent “Predicted Affinity” Audiences that expand a core audience to find potential customers that are predicted to have a strong response to advertising from a particular brand, by analyzing conversations across the social web.

# 4. Conclusion 
## 4.1 Potential Problems
It is still doubted that an artificial intelligence platform like FBLearner Flow can be used broadly across an organization, as most machine learning models can’t be generalized, as Schroepfer said. It seems quite common that sometimes there exists some gaps between the creators of a machine learning platform and those trying to build a product.

There is also a common concern about people's privacy. Facebook has developed the image recognition models with 98% accuracy to recognize human faces.  It is said that it could identify a person in one picture out of 800 million in less than five seconds. 

Facebook is meant to build a social network that can better anticipate what a user wants to see or experience. For example, if you have a bad day, he wants Facebook to show you humorous cat videos.  But currently, Facebook cannot yet really set up to optimize for that.

Ken Rudin pointed out that companies use big data to answer meaningless questions. For Facebook, a meaningful question is defined as one that leads to an answer that provides a basis for changing behavior. If you can’t imagine how the answer to a question would lead you to change your business practices, the question isn’t worth asking.”

## 4.2 Future development

Today, Facebook can conduct qualitative reviews of new products through different analysis methods such as sentiment analysis and image analysis with focus groups and direct user feedback. All of this have helped prevent “rocky” product launches. “Now it’s pretty rare for us to launch something where we didn’t understand how the change is better for people.”


Machine learning technologies are increasing efficiency of computer processing and enabling us to build a system at a scale never seen before.
They are helping Facebook expand the reach and capabilities of its social network without eroding the profits it generates. With a little luck, they’ll help us better learn how to live with machines.


# 5. Reference
> [reference]: http://fortune.com/facebook-machine-learning/ INSIDE FACEBOOK’S BIGGEST ARTIFICIAL INTELLIGENCE PROJECT EVER
> [reference]: http://andybromberg.com/sentiment-analysis-python/ Second Try: Sentiment Analysis in Python
> [reference]: https://www.crimsonhexagon.com/blog/machine-learning-social-media-analysis/ How Does Machine Learning Improve Social Media Analysis?
>[reference]: https://code.facebook.com/posts/181565595577955/introducing-deeptext-facebook-s-text-understanding-engine/ Introducing DeepText: Facebook's text understanding engine
>[reference]: https://medium.com/google-cloud/sentiment-analysis-of-comments-on-lhls-facebook-page-9db8b3a60eb3/ Sentiment Analysis of Comments on LHL’s Facebook Page
>[reference]: http://blog.aylien.com/sentiment-analysis-going-beyond-positive-and/ Sentiment Analysis: Going Beyond Positive and Negative