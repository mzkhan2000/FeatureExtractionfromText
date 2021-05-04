# FeatureExtractionfromText

In [1]:
# importing necessary libraries in jupyter notebook/ -Monir
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re, string # using to remove regular expression, special characters in the csv files/ -Monir
In [2]:
# importing csv files to the repositery as data frames/ -Monir
monir_df_fake = pd.read_csv("Fake.csv")
monir_df_true = pd.read_csv("True.csv")
In [3]:
monir_df_fake.head(5) # see how the data in the fake.cse file look like/ Monir
Out[3]:
title	text	subject	date
0	Donald Trump Sends Out Embarrassing New Year’...	Donald Trump just couldn t wish all Americans ...	News	December 31, 2017
1	Drunk Bragging Trump Staffer Started Russian ...	House Intelligence Committee Chairman Devin Nu...	News	December 31, 2017
2	Sheriff David Clarke Becomes An Internet Joke...	On Friday, it was revealed that former Milwauk...	News	December 30, 2017
3	Trump Is So Obsessed He Even Has Obama’s Name...	On Christmas day, Donald Trump announced that ...	News	December 29, 2017
4	Pope Francis Just Called Out Donald Trump Dur...	Pope Francis used his annual Christmas Day mes...	News	December 25, 2017
In [4]:
monir_df_true.head(5) # see how the data in the true.cse file look like/ Monir
Out[4]:
title	text	subject	date
0	As U.S. budget fight looms, Republicans flip t...	WASHINGTON (Reuters) - The head of a conservat...	politicsNews	December 31, 2017
1	U.S. military to accept transgender recruits o...	WASHINGTON (Reuters) - Transgender people will...	politicsNews	December 29, 2017
2	Senior U.S. Republican senator: 'Let Mr. Muell...	WASHINGTON (Reuters) - The special counsel inv...	politicsNews	December 31, 2017
3	FBI Russia probe helped by Australian diplomat...	WASHINGTON (Reuters) - Trump campaign adviser ...	politicsNews	December 30, 2017
4	Trump wants Postal Service to charge 'much mor...	SEATTLE/WASHINGTON (Reuters) - President Donal...	politicsNews	December 29, 2017
In [8]:
monir_df_fake.shape, monir_df_true.shape # find number of rows and columns/ Monir
Out[8]:
((23481, 4), (21417, 4))
In [16]:
monir_df_fake["class"] = 0
monir_df_true["class"] = 1
In [17]:
rdf_fake = monir_df_fake.sample(n=100) # generating random sample of 100 number of rows from fake dataframe/ Monir
rdf_true = monir_df_true.sample(n=100) # generating random sample of 100 number of rows from true dataframe/ Monir
In [18]:
rdf_fake.head(5)
Out[18]:
title	text	subject	date	class
12844	CROOKED HILLARY Slams The Bank She Took A $258...		politics	Oct 3, 2016	0
11890	TOBY KEITH Has AWESOME Response To Crybaby Att...	Country singer Toby Keith won t be bullied int...	politics	Jan 16, 2017	0
14194	OBAMABOT CONGRESSWOMAN: ISIS Beheadings Do Not...	U.S. Rep. Eddie Bernice Johnson (D-Dallas) arg...	politics	Mar 30, 2016	0
22969	US Hostage Survives Terrorist Ordeal in Syria ...	21st Century Wire says A small miracle a rar...	Middle-east	March 7, 2017	0
5796	#NeverTrump GOP Delegates: Party Brass Intimi...	Donald Trump is the Republican Party s presump...	News	June 20, 2016	0
In [19]:
rdf_true.head(5)
Out[19]:
title	text	subject	date	class
19686	Factbox: Reactions to speech by Myanmar's Suu ...	NAYPYITAW (Reuters) - Myanmar leader Aung San ...	worldnews	September 19, 2017	1
7137	Gun shops eye busy Black Friday despite Hillar...	SAN FRANCISCO (Reuters) - Christmas came early...	politicsNews	November 21, 2016	1
4992	Merkel meets Trump in clash of style and subst...	BERLIN (Reuters) - She is controlled and cauti...	politicsNews	March 13, 2017	1
891	U.S. lawmakers seek 'well-rounded biofuels pol...	WASHINGTON (Reuters) - A bipartisan group in t...	politicsNews	November 2, 2017	1
6945	Taiwan says contact with Trump agreed ahead of...	TAIPEI (Reuters) - The telephone call between ...	politicsNews	December 3, 2016	1
In [28]:
rdf_fake.shape, rdf_true.shape # find number of rows and columns/ Monir
Out[28]:
((100, 5), (100, 5))
In [26]:
monir_df_fake = monir_df_fake.drop(rdf_fake.index) # removing rows of random sample from fake dataframe/ Monir
monir_df_true = monir_df_true.drop(rdf_true.index)  # removing rows of random sample from true dataframe/ Monir
In [27]:
monir_df_fake.shape, monir_df_true.shape, rdf_fake.shape, rdf_true.shape # find  number of rows and columns/ Monir
Out[27]:
((23381, 5), (21317, 5), (100, 5), (100, 5))
In [29]:
df_for_manual_testing_sample = pd.concat([rdf_fake, rdf_true], axis=0) # Combining/merging DataFrames with Pandas/ Monir
In [30]:
df_for_manual_testing_sample.to_csv("manual_testing_sample.csv") # Saving as csv file/ Monir
In [31]:
df_merge_all = pd.concat([monir_df_fake, monir_df_true], axis=0) # Combining/merging DataFrames with Pandas/ Monir
In [33]:
df_merge_all.head(10)
Out[33]:
title	text	subject	date	class
0	Donald Trump Sends Out Embarrassing New Year’...	Donald Trump just couldn t wish all Americans ...	News	December 31, 2017	0
1	Drunk Bragging Trump Staffer Started Russian ...	House Intelligence Committee Chairman Devin Nu...	News	December 31, 2017	0
2	Sheriff David Clarke Becomes An Internet Joke...	On Friday, it was revealed that former Milwauk...	News	December 30, 2017	0
3	Trump Is So Obsessed He Even Has Obama’s Name...	On Christmas day, Donald Trump announced that ...	News	December 29, 2017	0
4	Pope Francis Just Called Out Donald Trump Dur...	Pope Francis used his annual Christmas Day mes...	News	December 25, 2017	0
5	Racist Alabama Cops Brutalize Black Boy While...	The number of cases of cops brutalizing and ki...	News	December 25, 2017	0
6	Fresh Off The Golf Course, Trump Lashes Out A...	Donald Trump spent a good portion of his day a...	News	December 23, 2017	0
7	Trump Said Some INSANELY Racist Stuff Inside ...	In the wake of yet another court decision that...	News	December 23, 2017	0
8	Former CIA Director Slams Trump Over UN Bully...	Many people have raised the alarm regarding th...	News	December 22, 2017	0
9	WATCH: Brand-New Pro-Trump Ad Features So Muc...	Just when you might have thought we d get a br...	News	December 21, 2017	0
In [46]:
df_to_use = df_merge_all.drop(["title", "subject", "date"], axis=1) #drop columns from pandas dataframe/ Monir
In [47]:
df_to_use.head(10)
Out[47]:
text	class
0	Donald Trump just couldn t wish all Americans ...	0
1	House Intelligence Committee Chairman Devin Nu...	0
2	On Friday, it was revealed that former Milwauk...	0
3	On Christmas day, Donald Trump announced that ...	0
4	Pope Francis used his annual Christmas Day mes...	0
5	The number of cases of cops brutalizing and ki...	0
6	Donald Trump spent a good portion of his day a...	0
7	In the wake of yet another court decision that...	0
8	Many people have raised the alarm regarding th...	0
9	Just when you might have thought we d get a br...	0
In [48]:
df_to_use = df_to_use.sample(frac=1) # Shuffle DataFrame rows/ Monir
In [49]:
df_to_use.head(10)
Out[49]:
text	class
20966	MOSCOW (Reuters) - Russian President Vladimir ...	1
15555	This video shows what happens when a large pop...	0
15220	BEIRUT (Reuters) - Saad al-Hariri, who resigne...	1
16305	BEIJING/TAIPEI (Reuters) - China urged the Uni...	1
7506	(Reuters) - In addition to picking the next U....	1
13658	GENEVA (Reuters) - The Syrian government has a...	1
1236	WASHINGTON (Reuters) - Advocates for Americans...	1
5275	Bill Clinton just watched his wife accept the ...	0
9511	The decision of actress Mila Kunis to make mon...	0
6430	WASHINGTON (Reuters) - U.S. Republican and Dem...	1
In [50]:
df_to_use.isnull().sum() # check out wheather null value available/ Monir
Out[50]:
text     0
class    0
dtype: int64
In [51]:
# define a function for removing unnecessary/special characters and return a lower case plain texts/ Monir
def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+ |www\.\S+', '', text)
    text = re.sub('<.*?>', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
In [52]:
df_to_use["text"]=df_to_use["text"].apply(word_drop) # apply the function
In [53]:
df_to_use.head(10)
Out[53]:
text	class
20966	moscow reuters russian president vladimir ...	1
15555	this video shows what happens when a large pop...	0
15220	beirut reuters saad al hariri who resigne...	1
16305	beijing taipei reuters china urged the uni...	1
7506	reuters in addition to picking the next u ...	1
13658	geneva reuters the syrian government has a...	1
1236	washington reuters advocates for americans...	1
5275	bill clinton just watched his wife accept the ...	0
9511	the decision of actress mila kunis to make mon...	0
6430	washington reuters u s republican and dem...	1
In [54]:
x = df_to_use["text"]
y = df_to_use["class"]
In [55]:
#Split a Dataframe into Train and Test Set where test size is 1/4 / Monir
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = .25)
In [56]:
# For Converting a collection of raw documents to a matrix of TF-IDF features/ Monir
from sklearn.feature_extraction.text import TfidfVectorizer
In [61]:
My_vec = TfidfVectorizer()
x_vec_train = My_vec.fit_transform(x_train)
x_vec_test = My_vec.transform(x_test)
classification using logistic regression
In [58]:
# Logistic Regression classifier/ Monir
from sklearn.linear_model import LogisticRegression
In [62]:
LR_C = LogisticRegression()
LR_C.fit(x_vec_train, y_train)
Out[62]:
LogisticRegression()
In [63]:
LR_C.score(x_vec_test, y_test)
Out[63]:
0.9870246085011186
In [64]:
Pred_LR_C = LR_C.predict(x_vec_test)
In [65]:
print(classification_report(y_test, Pred_LR_C))
              precision    recall  f1-score   support

           0       0.99      0.98      0.99      5778
           1       0.98      0.99      0.99      5397

    accuracy                           0.99     11175
   macro avg       0.99      0.99      0.99     11175
weighted avg       0.99      0.99      0.99     11175

classification with gradient boosting
In [66]:
from sklearn.ensemble import GradientBoostingClassifier
In [67]:
GB_C = GradientBoostingClassifier(random_state=0)
GB_C.fit(x_vec_train, y_train)
Out[67]:
GradientBoostingClassifier(random_state=0)
In [68]:
GB_C.score(x_vec_test, y_test)
Out[68]:
0.9957941834451901
In [69]:
Pred_GB_C = GB_C.predict(x_vec_test)
In [70]:
print(classification_report(y_test, Pred_GB_C))
              precision    recall  f1-score   support

           0       1.00      0.99      1.00      5778
           1       0.99      1.00      1.00      5397

    accuracy                           1.00     11175
   macro avg       1.00      1.00      1.00     11175
weighted avg       1.00      1.00      1.00     11175

classification with Randm Forest
In [71]:
from sklearn.ensemble import RandomForestClassifier
In [72]:
RF_C = RandomForestClassifier(random_state=0)
RF_C.fit(x_vec_train, y_train)
Out[72]:
RandomForestClassifier(random_state=0)
In [73]:
RF_C.score(x_vec_test, y_test)
Out[73]:
0.9896196868008948
In [74]:
Pred_RF_C = RF_C.predict(x_vec_test)
In [75]:
print(classification_report(y_test, Pred_RF_C))
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      5778
           1       0.99      0.99      0.99      5397

    accuracy                           0.99     11175
   macro avg       0.99      0.99      0.99     11175
weighted avg       0.99      0.99      0.99     11175

manually cheak out classifier prediction
In [77]:
def output_lebel(n):
    if n==0:
        return "This is a Fake news"
    elif n == 1:
        return "This is NOT a Fake news"
In [84]:
def manual_testing(news):
    testing_news={"text":[news]}
    new_def_test=pd.DataFrame(testing_news)
    new_def_test["text"]=new_def_test["text"].apply(word_drop)
    new_x_test = new_def_test["text"]
    new_xv_test = My_vec.transform(new_x_test)

    Pred_LR_C = LR_C.predict(new_xv_test)
    Pred_GB_C = GB_C.predict(new_xv_test)
    Pred_RF_C = RF_C.predict(new_xv_test)
    
    return print("\n\nLRC Prediction: {} \nGBC Preiction: {} \nRFC Preiction: {}".format(output_lebel(Pred_LR_C),
                                                                                         output_lebel(Pred_GB_C),
                                                                                         output_lebel(Pred_RF_C)))
In [85]:
news = str(input())
manual_testing(news)
Bill Maher laid into the GOP on Friday night for creating the racist mess that has taken over the party.During the program, the HBO host wasted no time summing up the Republican Party situation as he began the discussion with his panel of guests. The Republican Party is having an existential crisis. They are very upset that half their voters want to give nuclear weapons to a guy who gets into Twitter feuds with D-list celebrities. We understand that. I almost feel bad for them, except I really don t, because they brought this on themselves. They made a Faustian deal with the racist devil years ago and now those chickens are coming home to roost. He then read a statement made by House Speaker Paul Ryan, who said that any person who wants to be the GOP nominee must reject bigotry because  this party does not prey on people s prejudices. Maher immediately called bullshit, pointing out that if Republicans really believed that they wouldn t be pursuing voter ID laws that disenfranchise minority voters or jump to the defense of a police officer every time an unarmed black man is shot dead. Who are they kidding?  Maher asked.  This is the party they are and Trump is just the latest. Indeed, Donald Trump rallies have become nothing more than giant white supremacist rallies where Trump disparages people of color and orders his goons to beat up black protesters and remove them from the audience.Conservative author Matt Lewis admitted that the GOP has been dumbing down its voters but claimed that racism isn t as prevalent within the party as it seems. After claiming that the GOP presidential field is more diverse than the Democratic field, Maher noted that the two Hispanic candidates, Ted Cruz and Marco Rubio,  want to put other Hispanics on cattle carts and throw them out of the country. Maryland Rep. Donna Edwards also chimed in by pointing out that Republicans are starting to line up behind Trump despite objections to his racism, so they can t really have much of a problem with it.Maher also reminded Lewis that what Republicans are doing now isn t new. It was called the Southern Strategy when Nixon did it. Reagan started his campaign in Philadelphia, Mississippi. It s always been a winking campaign to get white poor people, mostly, who have a resentment to vote racially. Let s not pretend this is new. In the end, Maher said the GOP  big tent  is more like a house. The Republican establishment gave racists a room in their house and now the racists have taken over the house to the point where moderate Republicans are scared to go to the kitchen at night for a snack.Here s the video via YouTube. Featured image via screen capture


LRC Prediction: This is a Fake news 
GBC Preiction: This is a Fake news 
RFC Preiction: This is a Fake news
In [86]:
news = str(input())
manual_testing(news)
HELSINKI (Reuters) - Finland will leave the European Union and position itself as the Switzerland of the north to protect its independence if Laura Huhtasaari, the presidential candidate of the eurosceptic Finns Party has her way. She also told Reuters in an interview she wants to tighten immigration rules. Huhtasaari   dubbed  Finland s Marine Le Pen  after France s National Front leader   is a long-shot. But she believes she has a real chance in the January election as her party has taken a fresh start following its removal from the coalition government in June.  The rise in Europe of parties that are critical towards the EU and immigration is due to bad, unjust politics,  she said.     The role for Finland in the euro zone is the role of a loser and payer...  I do not want Finland to become a province of EU, Finns must stand up for Finland s interests.  The Finns Party, formerly called  True Finns , rose from obscurity during the euro zone debt crisis with an anti-EU platform, complicating the bloc s bailout talks with troubled states. It expanded into the second-biggest parliamentary party in 2015 and joined the government, but then saw its support drop due to compromises in the three-party coalition. This June, the party picked a new hard-line leadership and got kicked out of the government, while more than half of its lawmakers left the party and formed a new group to keep their government seats. Huhtasaari, 38, who was picked as deputy party leader in June, said voters were still confused after the split-up but that the party would eventually bounce back.  The game is really brutal. The biggest parties want us to disappear from the political map. No-one is in politics looking for friends.  The Finns party ranks fifth in polls with a support of 9 percent, down from 17.7 percent in 2015 parliamentary election, while the new  Blue Reform  group, which has five ministers, is backed by only 1-2 percent. Incumbent President Sauli Niinisto, who originally represented the centre-right NCP party, is widely expected to be elected for a second six-year term by a wide margin. A poll by Alma Media last week showed 64 percent of voters supporting Niinisto while 12 percent backed lawmaker Pekka Haavisto from the Greens. Huhtasaari, a first term lawmaker, was backed by 3 percent of those polled.  Things happen slowly when you re fighting against the hegemony... I still have time before the elections,  she said. Huhtasaari, who supports U.S. President Donald Trump and Britain s former UK Independence Party leader Nigel Farage, said the European eurosceptic movement was gradually strengthening despite a series of blows to anti-establishment parties. France s National Front and Italy s 5-Star Movement failed in attempts to win legislative and civic elections while UKIP won no seats in the British parliament, albeit that its goal of Brexit won a referendum.  Any change takes time, a step forward and step back... But the movement strengthens all the time,  she said, noting Austria s Freedom Party s strong performance in October elections. Markku Jokisipila, the director at the Center for Parliamentary Studies of the University of Turku, said Huhtasaari was unlikely to succeed in the Jan 28 election. Around 70 percent of Finns support EU membership and the centre-right government is committed to the euro.  There s no way around it, she is very inexperienced politically for this election,  Jokisipila said. He added that the Finns party had become more united after the June split-up, but that it was now too focused on its anti-immigration and anti-EU platforms to be able to increase support.  They will not disappear from the Finnish politics. The challenge is to broaden their profile... but they have also proved that they do have surprise potential.  


LRC Prediction: This is NOT a Fake news 
GBC Preiction: This is NOT a Fake news 
RFC Preiction: This is NOT a Fake news
