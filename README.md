# emoji2vec
![emoji](https://github.com/jiali-ms/emoji2vec/blob/master/logo.png)

A demo project to play word embedding and emoji with twitter data. Let's see how we can make a smarter emoji predictor.  
  
Know it or not, we have 1800+ emoji from standard Unicode. How we can find a interesting one from them? Traditional way is to search with key words of emoji description. We will use word embedding to find the best match with a context. The recommendations reflects real users habit from social media, you are guided with most knowledgeable emoji master :) 

##   Data 
The zip file in the data folder is a 1M sentences with emoji from Twitter about 2017-Jan. It is randomly selected set from a much bigger corpus. Unzip the corpus.txt directly into the data folder for training. 
## Training
Run [train.py](https://github.com/jiali-ms/emoji2vec/blob/master/train.py) file. Don't forget to set correct parameters like vector size, windows size, etc. It will dump a model and a raw text file for the embedding. 
