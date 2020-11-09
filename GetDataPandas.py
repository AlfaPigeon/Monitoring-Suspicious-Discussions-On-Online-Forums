import pandas as pd

reddit = pd.read_csv("/home/master/Projects/bil476/reddit.csv")

_input = 0
index = 101
pagenames = []
authors = []
titles = []
comments = []
targets = []
while (True):
    pagename = reddit['pagename'][index]
    author = reddit['author'][index]
    title = reddit['title'][index]
    comment = reddit['comment'][index]
    print("\n=============\n", "index: " + str(index),"\npagename:\n", pagename, "\nauthor:\n",
          author, "\ntitle:\n", title, "\ncomment:\n", comment)
    _input = input("toxic? 1/0\n")
    if(_input == "-1"):
        break
    pagenames.append(pagename)
    authors.append(author)
    titles.append(title)
    comments.append(comment)
    targets.append(_input)
    index += 1

df = pd.DataFrame({'pagename': pagenames,
                   'author': authors,
                   'title': titles,
                   'target': targets})
df.to_csv('labeled_reddit.csv', index=False) 
