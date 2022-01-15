import wikipedia

wikipedia.languages("en")
titles = {}
for i in range(1, 10):
    titles[i] = wikipedia.random(pages=200)
    print(i," ... 200 titles retrieved")

test = {}
dictlist = []
dictlist = titles.values()
for n in range(1,30000):
    test[n] = wikipedia.summary(title=dictlist[n],auto_suggest=True,  redirect=True).encode('utf-8')

daa = []
daa = test.values()


for i in range(0,29999):
    text_file = open("example_data/text//wiki_TEST2.txt", "a")
    text_file.write(daa[i])
    text_file.close()




# wikipedia.languages("de")
# titles = {}
# for i in range(500, 30000):

#     titles[i] = wikipedia.random(pages=page)
#     print(i,"titles retrieved")

# test = {}
# dictlist = []
# dictlist = titles.values()
# for n in range(1,30000):
#     test[n] = wikipedia.summary(title=dictlist[n],auto_suggest=True,  redirect=True).encode('utf-8')

# daa = []
# daa = test.values()

# for i in range(0,29999):
#     text_file = open("example_data/text//wiki_TEST2.txt", "a")
#     text_file.write(daa[i])
#     text_file.close()

