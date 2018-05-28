from model import *
from util import *

import matplotlib.pyplot as plt
import matplotlib.image as image
import os
import numpy as np
from sklearn.decomposition import PCA

d3_template = '''
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width">
  <title>JS Bin</title>
</head>
<body>
<!-- load D3js -->
<script src="//d3plus.org/js/d3.js"></script>
<!-- load D3plus after D3js -->
<script src="//d3plus.org/js/d3plus.js"></script>
<!-- create container element for visualization -->
<div id="viz"></div>
<script>
  // sample data array
  var sample_data = [
{}
  ]
  
    var attributes = [

  ]
  
  // instantiate d3plus
  var visualization = d3plus.viz()
    .container("#viz")  // container DIV to hold the visualization
    .data(sample_data)  // data to use with the visualization
    .size(20)
    .type("scatter")    // visualization type
    .id("name")         // key for which our data is unique on
    .x("x")         // key for x-axis
    .y("y")        // key for y-axis
    .attrs(attributes)
     .color("hex")
    .y({{"grid": false}})
    .x({{"grid": false}})
    .draw()             // finally, draw the visualization!
</script>
</body>
</html>
'''


model = EmojiModel()

# face emoji
face_emoji = '😀 😬 😁 😂 😃 😄 😅 😆 😇 😉 😊 🙂 🙃 😋 😌 😍 😘 😗 😙 😚 😜 😝 😛 🤑 🤓 😎 🤗 😏 😶 😐 😑 😒 🙄 🤔 😳 😞 😟 😠 😡 😔 😕 🙁 ☹ 😣 😖 😫 😩 😤 😮 😱 😨 😰 😯 😦 😧 😢 😥 😪 😓 😭 😵 😲 🤐 😷 🤒 🤕 😴'
face_emoji_list = face_emoji.split()

def emoji_words(top=1):
    """
    Get tuple of emoji and its top related words.
    :return:
    """
    result = []
    for emoji in face_emoji_list:
        # find most similar words
        similar_words = model.predict(emoji, '')
        similar_words = [x for x in similar_words if not is_emoji(x[0])][:top]
        print('{} {}'.format(emoji, similar_words))
        result.append((emoji, similar_words[:top]))

    return result

def dataset():
    #e_w = emoji_words()
    words = face_emoji_list
    # words = set(sum([[word[0] for word in x[1]] for x in e_w], [])) | set(face_emoji_list)
    return words, model.model[words]

def draw_cluster_matplot():
    # words, vectors
    words, vec = dataset()

    #words = ['a', 'b', 'c']
    # vec = np.array([[2.0, 1.0], [-1.0, -2.0], [0.5, -0.5]])

    # p = vec

    pca = PCA(n_components=2)
    p = pca.fit_transform(vec)

    fig, ax = plt.subplots()
    ax.axis([-10, 10, -7, 7])

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.xticks(range(-10, 10))
    plt.yticks(range(-7, 7))

    for i, word in enumerate(words):
        ax.annotate(word,
                    xy=(p[i][0], p[i][1]),
                    textcoords='offset points', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    fontname='Segoe UI Emoji',  # this is the param added
                    fontsize=30)

    plt.show()

def draw_cluster_d3():
    # words, vectors
    words, vec = dataset()

    pca = PCA(n_components=2)
    p = pca.fit_transform(vec)

    result = []
    for i, word in enumerate(words):
        result.append('{' + '"x": {}, "y": {}, "name": "{}"'.format(p[i][0], p[i][1], word) + '}')

    with open('d3_data.html', 'w', encoding='utf-8')as f:
        f.write(d3_template.format(',\n'.join(result)))

draw_cluster_d3()