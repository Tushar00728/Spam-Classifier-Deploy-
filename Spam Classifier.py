#!/usr/bin/env python
# coding: utf-8

# # Spam Classifier | NLP

# ### Spam classifier based on Bag of words 
# 
# 1) The data is taken from the UCI spam collection dataset : https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
# 
# 2) We will use Stemming 
# 
# 3) The proportion of ham and spam should be in equal proportion

# In[1]:


import pandas as pd


# In[2]:


messages = pd.read_csv('SMSSPAMCOLLECTION', sep = '\t', names = ["label", "message"]) #tab separator


# In[3]:


messages


# In[4]:


import re
import nltk
nltk.data.path.append("G:/Miniconda_projs/nltk_data/")


# ## Data Cleaning and preprocessing

# In[5]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]',' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    


# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000) #limit the words to top 5000 frequent words
X = cv.fit_transform(corpus).toarray()


# In[7]:


X


# In[8]:


X.shape #columns represent no. of words


# In[9]:


y = pd.get_dummies(messages["label"]) #converting text to dummies


# In[10]:


y


# In[11]:


y = y.drop("ham", axis = 1) #representing spam and ham using one column only


# In[12]:


y


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state =0)


# In[14]:


X_test.shape , X_train.shape


# In[35]:


X_test


# In[15]:


from sklearn.naive_bayes import MultinomialNB
spam_mod = MultinomialNB().fit(X_train,y_train)


# In[16]:


y_pred = spam_mod.predict(X_test)


# In[17]:


y_pred


# In[18]:


y_test


# In[19]:


from sklearn.metrics import confusion_matrix
conf_m = confusion_matrix(y_test,y_pred)


# In[20]:


conf_m


# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[22]:


fig, ax = plt.subplots(figsize=(3,3))
ax = sns.heatmap(conf_m,annot = True, cbar = False, fmt = ".2f")


# In[23]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)


# In[24]:


accuracy


# In[25]:


import pickle


# In[30]:


pickle.dump(spam_mod, open("model.pkl","wb"))


# In[31]:


model = pickle.load(open("model.pkl", "rb"))


# In[ ]:




