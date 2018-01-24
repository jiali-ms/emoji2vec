# emoji2vec
![emoji](https://github.com/jiali-ms/emoji2vec/blob/master/logo.png)

A demo project to play word embedding and emoji with twitter data. Let's see how we can make a smarter emoji predictor.  
  
Believe it or not, we have 1800+ emoji from standard Unicode. How to find one from them? Traditionally, we search key words of emoji description. We will use word embedding to find the best match with a context. The results reflects real users habit from social media. Now you are guided with most knowledgeable emoji master :)

Check the site http://nlpfun.com/emoji for a preview of what we can do next with the model!  

##   Data 
The zip file in the data folder is a 1M sentences with emoji from Twitter about 2017-Jan. It is randomly selected set from a much bigger corpus. Unzip the corpus.txt directly into the data folder for training. 
## Training
Run [train.py](https://github.com/jiali-ms/emoji2vec/blob/master/train.py) file. Don't forget to set correct parameters like vector size, windows size, etc. It will dump a model and a raw text file for the embedding.   

Install the gensim first with python 3.5
> pip install -r requirements.txt 
## Results
Let's run the [model.py](https://github.com/jiali-ms/emoji2vec/blob/master/model.py) and see the console output. If you ever played word2vec before, you know the answer for the famous play **'King' - 'man' + 'woman' = ?** . Let check the results with our simple twitter data.
> query + [**'king'**, **'woman'**] - [**'man'**]  
[(**'queen'**, 0.73), ('crown', 0.7), ('goddess', 0.7), ('princess', 0.7), ('actress', 0.69)]

>query + [**'china'**, **'tokyo'**] - [**'beijing'**]  
[(**'japan'**, 0.73), ('theatre', 0.69), ('europe', 0.68), ('nyc', 0.67), ('nuttiness', 0.67)]

>query + [**'dog'**, **'cats'**] - [**'dogs'**]  
[(**'cat'**, 0.92), ('kitten', 0.75), ('puppy', 0.73), ('coworker', 0.71), ('mom', 0.69)]

Finally, let's see how well we can find emoji by key words!
> query + [**'cat'**] - []  
[('🐱', 0.66), ('🐶', 0.5), ('🐈', 0.47), ('🐰', 0.42), ('🐕', 0.4), ('🐭', 0.39)]

>query + [**'happy'**, **'new'**, **'year'**] - []  
[('🎉', 0.56), ('🎄', 0.38), ('🏆', 0.38), ('🍾', 0.37), ('🎁', 0.36)]

>query + [**'king'**, **'woman'**] - [**'man'**]  
[('👑', 0.63), ('👸', 0.59), ('🦁', 0.56)]

>query + ['💗'] - []  
[('💓', 0.94), ('💜', 0.92), ('💖', 0.92), ('💘', 0.91), ('💞', 0.91), ('💕', 0.91), ('💛', 0.81), ('💙', 0.81), ('💟', 0.79), ('💝', 0.72)]


