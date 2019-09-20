#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pandas as pd
import re ,nltk
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import GridSearchCV,cross_val_score,cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.metrics import make_scorer
import warnings
warnings.filterwarnings("ignore")


# In[82]:


def Visualize(tweet,Name,Class):
    Plot=sns.distplot(Class,kde=False,bins=3,rug=False,color="#07438C")
    Plot.set(xlabel="Negative          Neutral            Positive",
             ylabel="No of tweets",
             title="tweets of {} ".format(Name)
    )
    Plot.plot()
    
    StopWords=set(STOPWORDS)
    all_words = ' '.join([text for text in tweet])
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

    plt.figure(figsize=(12, 10))
    plt.axis('off')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.show()


# In[83]:


def hashtag(tweets):
    hashtag=[]
    for tweet in tweets:
        ht=re.findall(r'#(\w+)',tweet)
        if ht: hashtag.append(ht)
    return hashtag

def plot(HashTag):
    a = nltk.FreqDist(HashTag)
    d = pd.DataFrame({'Hashtag': list(a.keys()),
                      'Count': list(a.values())})
    # selecting top 10 most frequent hashtags
    
    d = d.nlargest(columns="Count", n = 10) 
    plt.figure(figsize=(16,5))
    ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
    ax.set(ylabel = 'Count')
    plt.show()


# In[84]:


def Clean(Tweets):
    from nltk.tokenize import WordPunctTokenizer
    pattern=re.compile(r'(https|http):?//[a-zA-Z0-9./]+|<.*?>|@\w+|RT @\w+')
    Stopwords={sw for sw in nltk.corpus.stopwords.words('english')}
    processed=[]
    
    for tweet in Tweets:
        # remove html tags,URL links,twitter handles
        tweet=pattern.sub('',tweet)
        
        
        #remove all special characters (except #)
        tweet=re.sub(r'[^a-zA-Z0-9# ]'," ",tweet).lower()
        
        
        # tokenize to remove extra space and remove stopwords
        tweet=tweet.split()
        tweet=' '.join(([w for w in tweet if not w in Stopwords]))
        
        #Convert each word to its root word
        Root=nltk.stem.PorterStemmer()
        Tweet_root=[Root.stem(w) for w in tweet.split()]
        tweet=tweet=str(' '.join(Tweet_root))
        
        processed.append(tweet)
        
        
    return processed


# In[77]:


def Parameter_Tuning(parameters,model,X,y):
    
    gs_clf=GridSearchCV(model,parameters,cv=10,iid=False,n_jobs=-1)
    GS_clf=gs_clf.fit(X,y)
    
    
    print("Best Score for the model--->{0}".format(GS_clf.best_score_))
    print("Best Parameters are...")
    print(GS_clf.best_params_)
    
#     print(scores.keys()) 
#     ['test_Precision', 'train_F1_score', 'fit_time', 'train_Precision', 'test_Recall', 'score_time', 'train_Recall', 'test_F1_score']
    metric=['test_Precision','test_Recall','test_F1_score']

    for m in metric:
        
        #positve
        scoring={
            'F1_score':make_scorer(f1_score,labels=[1],average=None),
            'Precision':make_scorer(precision_score,labels=[1],average=None),
            'Recall':make_scorer(recall_score,labels=[1],average=None)
        }
        
        scores=cross_validate(GS_clf.best_estimator_,X,y,cv=10,scoring=scoring)
        print("{} for positive class:{}".format(m,scores[m].mean()))
        
        #negative
        scoring={
            'F1_score':make_scorer(f1_score,labels=[-1],average=None),
            'Precision':make_scorer(precision_score,labels=[-1],average=None),
            'Recall':make_scorer(recall_score,labels=[-1],average=None)
        }

        scores=cross_validate(GS_clf.best_estimator_,X,y,cv=10,scoring=scoring)
        print("{} for negative class:{}".format(m,scores[m].mean()))
    
        #neutral
        scoring={
            'F1_score':make_scorer(f1_score,labels=[0],average=None),
            'Precision':make_scorer(precision_score,labels=[0],average=None),
            'Recall':make_scorer(recall_score,labels=[0],average=None)
        }

        scores=cross_validate(GS_clf.best_estimator_,X,y,cv=10,scoring=scoring)
        print("{} for neutral class:{}".format(m,scores[m].mean()))
    return GS_clf
    
    
    
    
    


# In[85]:


def Test(model,X_test,X_Test_ID,x_test_name):
    y_pred=model.predict(X_test)
    Filename='Sneha_Shet_Parth_Desai_'+x_test_name+'.txt'
    with open (Filename,'w') as result_file:
        for i,y in zip(X_Test_ID,y_pred):
            result_file.write(str(i)+';;'+str(y)+'\n')


# In[86]:


def main():
    df_Obama_tweets=df_Obama_tweets=pd.read_excel('trainingObamaRomneytweets.xlsx',sheet_name='Obama')
    df_Romney_tweets=pd.read_excel('trainingObamaRomneytweets.xlsx',sheet_name='Romney')

    # retains first 4 columns
    df_Obama_tweets=df_Obama_tweets[1:].dropna(axis=1,how='all')
    df_Romney_tweets=df_Romney_tweets[1:].dropna(axis=1,how='all')



    #rename the colums
    df_Obama_tweets.columns=['Date','Time','Tweets','Class']
    df_Romney_tweets.columns=['Date','Time','Tweets','Class']
    df_Obama_tweets.index=range(1,len(df_Obama_tweets)+1)
    df_Romney_tweets.index=range(1,len(df_Romney_tweets)+1)


    #drop mixed and !!! class for now(we may consider it later)
    df_Romney_tweets=df_Romney_tweets[(df_Romney_tweets['Class'].isin((0,1,-1)))]
    df_Obama_tweets=df_Obama_tweets[(df_Obama_tweets['Class'].isin((0,1,-1)))]

    #drop empty tweet rows
    df_Obama_tweets.dropna(subset=['Tweets'],inplace=True)
    df_Romney_tweets.dropna(subset=['Tweets'],inplace=True)

    #shuffle the dataframe to avoid bias
    df_Obama_tweets=shuffle(df_Obama_tweets).reset_index(drop=True)
    df_Romney_tweets=shuffle(df_Romney_tweets).reset_index(drop=True)

    #raw tweet and class
    Obama_tweet=df_Obama_tweets['Tweets'].tolist()
    Obama_class=df_Obama_tweets['Class'].tolist()

    Romney_tweet=df_Romney_tweets['Tweets'].tolist()
    Romney_class=df_Romney_tweets['Class'].tolist()


    # preprocess the tweets
    Xtrain_Obama=Clean(Obama_tweet)
    Xtrain_Romney=Clean(Romney_tweet)

    #Visualize
    Visualize(Xtrain_Obama,"Obama",Obama_class)
    Visualize(Xtrain_Romney,"Romney",Romney_class)


    #Analysing Hashtags

    #Positive tweets
    positive_Romney=df_Romney_tweets.loc[df_Romney_tweets['Class']==1]['Tweets']
    positive_Obama=df_Obama_tweets.loc[df_Obama_tweets['Class']==1]['Tweets']
    positive_raw=positive_Romney.append(positive_Obama)

    positive=Clean(positive_raw)
    HT_positive=sum(hashtag(positive),[])

    plot(HT_positive)

    #Negative tweets
    Negative_Romney=df_Romney_tweets.loc[df_Romney_tweets['Class']==-1]['Tweets']
    Negative_Obama=df_Obama_tweets.loc[df_Obama_tweets['Class']==-1]['Tweets']
    Negative_raw=Negative_Romney.append(Negative_Obama)

    Negative=Clean(Negative_raw)
    HT_Negative=sum(hashtag(Negative),[])

    plot(HT_Negative)
    
    ############################# Multinomail NB############################################

    #Vectorize the tweet and initialize the classifiers
    tweet_clf=Pipeline([
        ('count_vect',CountVectorizer(max_df=.7)),
        ('vect_tfidf',TfidfTransformer()),
        ('clf',MultinomialNB()),
    ])

    #tuning the parameters
    parameters={
        'count_vect__ngram_range':[(1,1),(1,2)],
        'count_vect__max_df':(.5,.6,.8,.9),
        'vect_tfidf__use_idf':(True,False),
        'clf__alpha':(1e-2,1e-3,1e-4)
    }

    model_NB_obama=Parameter_Tuning(parameters,tweet_clf,Xtrain_Obama,Obama_class)
    model_NB_romney=Parameter_Tuning(parameters,tweet_clf,Xtrain_Romney,Romney_class)
    

    
    ####################Linear SVM#############################################################
    
    tweet_clf=Pipeline([
        ('count_vect',CountVectorizer(max_df=.7)),
        ('vect_tfidf',TfidfTransformer()),
        ('clf',LinearSVC(loss='hinge', penalty='l2', random_state=42,
                              max_iter=5)),
    ])


    #tuning the parameters
    parameters={
        'count_vect__ngram_range':[(1,1),(1,2)],
        'count_vect__max_df':(.5,.6,.8,.9),
        'vect_tfidf__use_idf':(True,False),
        'clf__C':(1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3)
    }

    Model_SVM_obama=Parameter_Tuning(parameters,tweet_clf,Xtrain_Obama,Obama_class)
    Model_SVM_romney=Parameter_Tuning(parameters,tweet_clf,Xtrain_Romney,Romney_class)
    
    
    ##########################################Logistic Regression##################################

    tweet_clf=Pipeline([
        ('count_vect',CountVectorizer(max_df=.7)),
        ('vect_tfidf',TfidfTransformer()),
        ('clf',LogisticRegression(random_state=0)),
    ])


    #tuning the parameters
    parameters={
        'count_vect__ngram_range':[(1,1),(1,2)],
        'count_vect__max_df':(.5,.6,.8,.9),
        'vect_tfidf__use_idf':(True,False),
        'clf__C':(1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3),
        'clf__penalty': ['l2']
    }

    Model_Logistic_obama=Parameter_Tuning(parameters,tweet_clf,Xtrain_Obama,Obama_class)
    Model_Logistic_romney=Parameter_Tuning(parameters,tweet_clf,Xtrain_Romney,Romney_class)
    

    #################################Decision tree#####################################
    
    tweet_clf=Pipeline([
        ('count_vect',CountVectorizer(max_df=.7)),
        ('vect_tfidf',TfidfTransformer()),
        ('clf',DecisionTreeClassifier()),
    ])


    #tuning the parameters
    parameters={
        'count_vect__ngram_range':[(1, 1), (1, 2)],
        'count_vect__max_df':(.6,.8,.9),
        'vect_tfidf__use_idf':(True,False),
        'clf__min_impurity_decrease':[(1e-05), (1e-04), (1e-03)], 
        'clf__min_samples_leaf':[0.1, 0.5, 1],
        'clf__min_samples_split':[2, 3, 4, 5], 
        'clf__min_weight_fraction_leaf':[0.0, 0.1, 0.5],
        'clf__presort':[False]
    }

    Model_DT_obama=Parameter_Tuning(parameters,tweet_clf,Xtrain_Obama,Obama_class)
    Model_DT_romney=Parameter_Tuning(parameters,tweet_clf,Xtrain_Romney,Romney_class)
    

    ################################testing#############################################
    
    X_Test_Tweets_Romney=pd.read_csv("Romney_Test_dataset_NO_Label.csv",encoding = "ISO-8859-1")
    X_Test_Tweets_Obama=pd.read_csv("Obama_Test_dataset_NO_Label.csv",encoding = "ISO-8859-1")                                  
    X_Test_Tweets_Romney=X_Test_Tweets_Romney.dropna(axis=1,how='all')
    X_Test_Tweets_Obama=X_Test_Tweets_Obama.dropna(axis=1,how='all')                                 


    #rename the colums
    X_Test_Tweets_Romney.columns=['TweetID','Tweets']
    X_Test_Tweets_Romney.index=range(1,len(X_Test_Tweets_Romney)+1)

    X_Test_Tweets_Obama.columns=['TweetID','Tweets']
    X_Test_Tweets_Obama.index=range(1,len(X_Test_Tweets_Obama)+1)

    #drop empty tweet rows
    X_Test_Tweets_Obama.dropna(subset=['Tweets'],inplace=True)
    
    X_Test_tweet_Obama=X_Test_Tweets_Obama['Tweets'].tolist()
    X_Test_ID_Obama=X_Test_Tweets_Obama['TweetID'].tolist()

    X_Test_Tweets_Romney.dropna(subset=['Tweets'],inplace=True)
    
    X_Test_tweet_Romney=X_Test_Tweets_Romney['Tweets'].tolist()
    X_Test_ID_Romney=X_Test_Tweets_Romney['TweetID'].tolist()




    #Testing
    Test(Model_SVM_obama,X_Test_tweet_Obama,X_Test_ID_Obama,'Obama')
    Test(Model_SVM_romney,X_Test_tweet_Romney,X_Test_ID_Romney,'Romney')

    
    


# In[80]:


if __name__ == "__main__":
    main()


# In[ ]:




