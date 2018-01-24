# emoji2vec
![emoji](https://github.com/jiali-ms/emoji2vec/blob/master/logo.png)

A demo project to play word embedding and emoji with twitter data. Let's see how we can make a smarter emoji predictor.  
  
Believe it or not, we have 1800+ emoji from standard Unicode. How to find one from them? Traditionally, we search key words of emoji description. We will use word embedding to find the best match with a context. The results reflects real users habit from social media. Now you are guided with most knowledgeable emoji master :)

Check the site https://nlpfun.com/emoji for a preview of what we can do next with the model!  

##   Data 
The zip file in the data folder is a 1M sentences with emoji from Twitter about 2017-Jan. It is randomly selected set from a much bigger corpus. Unzip the corpus.txt directly into the data folder for training. 
## Training
Run [train.py](https://github.com/jiali-ms/emoji2vec/blob/master/train.py) file. Don't forget to set correct parameters like vector size, windows size, etc. It will dump a model and a raw text file for the embedding. 
