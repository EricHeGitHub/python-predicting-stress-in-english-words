# python-predicting-stress-in-english-words

1.	Introduction

In this project, a classifier used to predict stress of English words is created to classify English words with up to four phonemes and only one stress being marked as 1 in the training dataset. In order to solve this problem, a decision tree is utilised by implementing sklearn.tree.DecisionTreeClassifier with limited height in order to prevent overfitting. The following section will illustrate features selected, ways to implement, experiment and improve the classifier and the result analysis.

2.	Dictionary Computation

Before coming up with the features, I calculated some probability dictionaries from the training data in order to provide some information for the tree to learn.

1)	For each possible vowel, I calculate the probability of this vowel to be a stress and store such information in a dictionary, which can be used to calculate vow_P1, vow_P2, vow_P3 and vow_P4
2)	For each possible consonant, I calculate the probability of the consonant, such that the following vowel is a stress, which can be used to calculate const_P1, const_P2, const_P3 and const_P4.
3)	For each possible combination of consonant and vowel in the training set, I calculated the probability of such appearance in the training data set and store it in the dictionary, which can be used to calculate CV_P1, CV_P2, CV_P3 and CV_P4
4)	For each possible combination of the last vowel and its one following consonant in a word, I calculated the possibility of such appearance in the training set and store such tail pattern in a dictionary, which is used for VC_tail
5)	For each possible combination of the last vowel and its two following consonants in a word, I calculated the possibility of such appearance in the training set and store such tail pattern in a dictionary, which is used for VCC_tail
6)	For each prefix, I created three dictionaries:
a.	prefix_list2 = ['CO', 'DE','EM', 'EN', 'EX','IL', 'IM', 'IN', 'IR','RE','UN','DE', 'EX']
b.	prefix_list3 = ['DIS','EPI','MID', 'MIS','NON','PRE', 'SUB','TRI', 'UNI','COM', 'CON','SUB', 'TRI']
c.	prefix_list4 = ['ANTE', 'ANTI', 'CIRCUM','EXTRA','FORE', 'HYPER', 'INFRA', 'INTER', 'INTRA','MACRO', 'MONO', 'OMNI', 'POST','SUPER', 'THERM', 'TRANS', 'CONTRA','HOMO', 'MAGN', 'PARA','TRANS']
If a word has a prefix in prefix_list2, then the feature ‘prefix’ is equal to 2.
If a word has a prefix in prefix_list3, then the feature ‘prefix’ is equal to 3.
If a word has a prefix in prefix_list4, then the feature ‘prefix’ is equal to 4.
otherwise, the feature ‘prefix’ is equal to 1.

7)	For each suffix, I created three dictionaries:
a.	suffix_list2 = ['AL','AR','AL','ER','ED','EE','EN','CE','EL','IC','FY','LY','LO','NE','NO','ON','OR','NA','TH','TY','US']
b.	suffix_list3 = ['ADE','AGE','ACY','ARD','ANT','ARY','ATE','AUX','CHE','DOX','DOM','EER','ERT','FUL','FUS','EST','KIE','IAL','IAN','ILE','IFY','ILY','INE','ING','ION','OUS','ISE','ISH','ISM','IST','ITE','ITY','IVE','LOG','IZE','LET','MAN','NEY','OID','OUS','SIR','OMA','ORY','TON']
c.	suffix_list4 = ['ABLE','ANCE','ATIVE','CIDE','CRACY','CRAT','CULE','CHIN','DATO','DIAN','EMIA','ENCE','ESQUE','ETIC','ETTE','GRAM','GRAMY','HOOD','IASIS','IBLE','ICAL','IOUS','ITIS','IZATION','LESS','LIKE','LING','LONGER','LOGIST','LLOW','MENT','NESS','OLOGY','ONYM','OPIA','OPSY','OSIS','OSTOMY','PATH','PATHY','PHILE','PHONE','PHYTE','PLEGIA','PLEGIC','PNEA','SCOPY','SCOPE','SCRIBE','SCRIPT','SECT','SHIP','SION','SOME','SOPHY','SOPHIC','SKI','SSUS','TION','TOME','TOMY','TROPHY','TUDE','ULAR','UOUS','URE','WARD','WORTH','WARE','WISE', 'WARDS']

If a word has a suffix in suffix_list2, then the feature ‘suffix’ is equal to 2.
If a word has a suffix in suffix_list3, then the feature ‘suffix is equal to 3.
If a word has a suffix in suffix_list4, then the feature ‘suffix is equal to 4.
otherwise, the feature ‘suffix is equal to 1.

3.	Feature Selection

Most selected features are based on the probabilistic model which gives rise to the probability of features being significant in the classifier and instances. All features along with the explanation of their roles in the mode are given in the table below. There are 20 features in total.

1	length	The length of the word	Normally the longer a word is, it is possible that the stress is placed in latter vowels
2	Constant_count	The number of Constant in the word.	Usually, more consonants mean the length might be longer and might contain more vowels. The stress is possible to be placed in latter vowels.
3	Vowel_count	The probability of the length of a word learned from the training set.	Since there are up to four vowels in a word, the number of vowels can only be 1,2,3 and 4. P1, P2, P3 and P4 denote the probability of the number of phonemes being the length of 1,2,3 and 4. Still, normally, if there are more vowels, latter vowels are likely to be stressed.
4	vow_P1	The probability of the first vowel to be the stress of the word learned from the training set.	After learning from the training set, if Pc1 is higher, it is more likely that this position is a stress.
5	const_P1	The probability of the consonant before the first vowel, such that the vowel after this consonant is a stress.	Due to the definition of the probability of such consonant, the higher such probability is, the vowel right after it is more likely to be a stress.
6	vow_P2	The definition is the same as vow_P1, with the only difference being that the probability is defined for the second vowel.	The same explanation as vow_P1 and the only difference is that this explanation applies to the second vowel.
7	const_P2	The definition is the same as const_P1 with the only difference being that the probability is defined for the consonant before the second vowel being a stress.	The same explanation as const_P1 and the only difference is that this explanation applies to the consonant before the second stressed vowel.
8	vow_P3	The definition is the same as vow_P1, with the only difference being that the probability is defined for the third vowel.	The same explanation as vow_P3 and the only difference is that this explanation applies to the third vowel. Set to 0 if no third vowel exists.
9	const_P3	The definition is the same as const_P1, with the only difference being that the probability is defined for the consonant before the third vowel being a stress.	The same explanation as const_P1 and the only difference is that this explanation applies to the consonant before the third stressed vowel. Set to 0 if no third vowel exists.
10	vow_P4	The definition is the same as vow_P1, with the only difference being that the probability is defined for the forth vowel.	The same explanation as vow_P1 and the only difference is that this explanation applies to the forth vowel. Set to 0 if no forth vowel exists.
11	const_P4	The definition is the same as const_P1, with the only difference being that the probability is defined for the consonant before the forth vowel being a stress.	The same explanation as const_P1 and the only difference is that this explanation applies to the consonant before the third stressed vowel. Set to 0 if no forth vowel exists.
12	CV_P1	The probability of the combination of a consonant and a vowel where the vowel is the first vowel in the word and the consonant is the one before this. The probability is learned from the training set.	If a combination of a consonant and a vowel is likely to be a stress of the word, high probability of such combination in a test case indicates that it might follow that same stress rule as the ones of the same pattern in the training data set.
13	CV_P2	Similar as CV_P1 but for the second vowel position.	The same significance as CV_P1
14	CV_P3	Similar as CV_P1 but for the third vowel position. Set to 0 if no such combination exists during the test.	The same significance as CV_P1
15	CV_P4	Similar as CV_P1 but for the forth vowel position. Set to 0 if no such combination exists during the test.	The same significance as CV_P1
16	last_pronoun	The last pronunciation of the word	Normally, last pronunciation is closely related to the position of the stress, e.g. with the last pronunciation of ‘tion’, it is highly possible that the stress is located in the third or the forth position.
17	VC_tail	The probability of the combination of the last vowel and the consonant that follows such vowel. If the last vowel does not have a following consonant, the consonant is set to empty. In the test case, the probability is 0 if new combination comes to place	The last vowel and the following consonant gives information about the pronunciation pattern of the last part of a word, which is decisive in word type, i.e. noun or adjective. This is crucial to the position of stress in words.
18	VCC_tail	VC_tail only considers the consonant right after the last vowel. However, in some cases, there might be two consonants after the last vowel, the probability such combination is also included as a feature. Also, it is set to 0 if in the test case, new combination comes up.	It provides the similar information as VC_tail, only with a more comprehensive and inclusive description. Such information is also related to the type of words, i.e. noun, adjective or others.
19	prefix	The length of prefix of a word	The values are 1, 2, 3, 4 which represents the length of the prefix in the prefix dictionary
20	suffix	The length of suffix of a word	The values are 1, 2, 3, 4 which represents the length of the suffix in the suffix dictionary

4.	Experiment and Improvement

Cross-validation is used to test the performance of the classifier. The sklearn packet - from sklearn.cross_validation import train_test_split- is used to split the total training set of 50,000 items into 80% training set and 20% of testing set, the proportion of which is closest to the feedback given by the project online testing system.

The training and testing set is separated by the following sentence (commented for final submission)
train, test = train_test_split(instance_with_label, test_size = 0.2,random_state=42)
Also the test is conducted by 
X_test, y_test = list(zip(*test))
X_test = vectorizer.fit_transform(X_test)
y_pred_class = tree.predict(X_test)
Where X_test is the test instance and  y_test is the gorund truth of the test data while y_pred_class is the prediction of classifier of the classifier.

In order to improve my classifier, I try to optimise two key parameters of the tree classifier, which is defined as follows:
tree = DecisionTreeClassifier(criterion = "gini",max_depth=16)

The two key parameters are ‘criterion’ and ‘max_depth’. ‘criterion’ =  ‘gini’ or ‘entropy’ and depth can be any positive integer number.

The following table records the process of experiment and improvement of my tree classifier with different parameter value selection. To prevent overfitting, I limited the height of the tree to 20.
No.	criterion	max_depth	Cross-validation F1 score
1	Gini	1	0.3527
2	Gini	2	0.5000
3	Gini	3	0.5479
4	Gini	4	0.5623
5	Gini	5	0.5699
6	Gini	6	0.5948
7	Gini	7	0.6003
8	Gini	8	0.6998
9	Gini	9	0.6923
10	Gini	10	0.7067
11	Gini	11	0.7145
12	Gini	12	0.6904
13	Gini	13	0.7147
14	Gini	14	0.7058
15	Gini	15	0.6817
16	Gini	16	0.6964
17	Gini	17	0.6946
18	Gini	18	0.6745
19	Gini	19	0.6865
20	Gini	20	0.6689
21	Gini	25	0.6717
22	Gini	30	0.6732
23	Gini	35	0.6920
24	Gini	50	0.6923
No.	criterion	max_depth	Cross-validation F1 score
1	Entropy	1	0.2045
2	Entropy	2	0.3755
3	Entropy	3	0.5246
4	Entropy	4	0.5630
5	Entropy	5	0.5752
6	Entropy	6	0.5863
7	Entropy	7	0.6043
8	Entropy	8	0.6908
9	Entropy	9	0.7042
10	Entropy	10	0.6609
11	Entropy	11	0.7107
12	Entropy	12	0.6830
13	Entropy	13	0.6869
14	Entropy	14	0.6922
15	Entropy	15	0.6881
16	Entropy	16	0.6920
17	Entropy	17	0.6752
18	Entropy	18	0.6837
19	Entropy	19	0.6799
20	Entropy	20	0.6813
21	Entropy	25	0.6803
22	Entropy	30	0.6802
23	Entropy	35	0.6608
24	Entropy	50	0.6824
If the table above is transferred to a graph, the following graph can be obtained with x-axis being the depth of the tree and y-axis being the F1 score of cross-validation. 
Based on the two lines of trees using Gini and Entropy criteria respectively, it can be concluded that when the depth is larger than 10, both Gini tree and Entropy tree reach a relatively stable accuracy. As a result, in order to create generally stable result, I choose the max depth of 13 using Gini criterion because Gini focuses more on how much a randomly chosen instance will be misclassified if the tree is grown by a certain attribute and the max depth of 13 gives the highest F1 score.

5.	Conclusion and Performance Evaluation

As the tree and features are implemented as described above, to evaluate the performance, I conduct 10 replications of the cross-validation and results are as follows:

No. of Replication	F1 Score
1	0.7084
2	0.7123
3	0.6843
4	0.6857
5	0.7045
6	0.7132
7	0.7084
8	0.7094
9	0.7069
10	0.6916
Mean	0.70247

Conclusion: The tree implemented with the features and parameters above can achieve an accuracy of 0.70247 in average.
