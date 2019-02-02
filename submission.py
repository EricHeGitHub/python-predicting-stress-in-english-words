## import modules here 
import helper
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from numpy import asarray
import pickle
#from sklearn.cross_validation import train_test_split
from sklearn import metrics
#from sklearn.metrics import f1_score
################# training #################
 
def partial_string_division(word):
    pronunciations_int = word.copy()
    substrings = []
    consnt_before_v = ' '
    previous_pronunciation = ' '
    for pronunciation_index in range(len(pronunciations_int)):
        if pronunciations_int[pronunciation_index ][-1].isalpha():           
            consnt_before_v  = pronunciations_int[pronunciation_index]
            previous_pronunciation = pronunciations_int[pronunciation_index]

        else: 
            vowel_current = pronunciations_int[pronunciation_index ][:-1]
            
            if isVowel(previous_pronunciation )==1:
                consnt_before_v  = ' '

            substr = consnt_before_v + vowel_current
            substrings.append(substr)            
            previous_pronunciation = vowel_current                                
    for i in range(len(substrings)):
        substrings[i]= ''.join([j for j in substrings[i] if j!= ' '])
    while len(substrings)<4:
        substrings.append(' ')
    return substrings

def substring_dict(data):
    substring_dictionary = {' ':0}
    count = 0
    for line in data:
        pronunciationString = line.split(':')[1]
        pronunciations_int = pronunciationString.split(' ')
        substrings = partial_string_division(pronunciations_int)
        for sub in substrings:
            count +=1
            if sub not in substring_dictionary.keys():
                substring_dictionary[sub] = 1
            else:
                substring_dictionary[sub] += 1

    for key in substring_dictionary.keys():
        number = substring_dictionary[key]
        substring_dictionary[key] = float(number )/ count
    return substring_dictionary

def count_vowel_number(word):
    number = 0
    for pronunciation in word:
        if isVowel(pronunciation) ==1:
            number +=1
    return number

def prob_proun_number(data):
    vowels2 = []        
    vowels3 = []
    vowels4 = []
    for line in data:
        pronunciationString = line.split(':')[1]
        pronunciationString_no_int = ''.join([i for i in pronunciationString if not i.isdigit()])
        count_vowel = count_vowel_number(pronunciationString_no_int.split(' '))
        if count_vowel == 2:
            vowels2.append(line)
        elif count_vowel == 3:
            vowels3.append(line)
        elif count_vowel == 4 :
            vowels4.append(line)
    pDictionary = {}
    prob2 = probDistribution(vowels2)
    prob3 = probDistribution(vowels3)
    prob4 = probDistribution(vowels4)
    pDictionary[2]=prob2
    pDictionary[3]=prob3
    pDictionary[4]=prob4
    return pDictionary


def VC_search(pronunciations):
    vowel_count = 0
    number = count_vowel_number(pronunciations)
    VC_string = ''
    following_c = ''
    for pronunciation_index in range(len(pronunciations)):
        if isVowel(pronunciations[pronunciation_index]) == 1:
            vowel_count +=1
            if vowel_count == number:
                last_vowel = pronunciations[pronunciation_index]
                if pronunciation_index == len(pronunciations) -1:
                    following_c = ''
                else:
                    following_c = pronunciations[pronunciation_index+1]
                VC_string = last_vowel + following_c
    return VC_string

def VC_Dict_Construct(data):
    substring_dictionary = {}
    count = 0
    for line in data:
        pronunciationString = line.split(':')[1]
        pronunciationString_no_int = ''.join([i for i in pronunciationString if not i.isdigit()])
        pronunciations_no_int = pronunciationString_no_int.split(' ')
        VC_string = VC_search(pronunciations_no_int)
        count +=1
        if VC_string not in substring_dictionary.keys():
            substring_dictionary[VC_string] = 1
        else:
            substring_dictionary[VC_string] += 1

    for key in substring_dictionary.keys():
        substring_dictionary[key] = float(substring_dictionary[key])/ count
 
    return substring_dictionary   

def VCC_search(pronunciations):
    vowel_count = 0
    number = count_vowel_number(pronunciations)
    VC_string = ''
    following_c = ''
    for pronunciation_index in range(len(pronunciations)):
        if isVowel(pronunciations[pronunciation_index]) == 1:
            vowel_count +=1
            if vowel_count == number:
                last_vowel = pronunciations[pronunciation_index]
                if pronunciation_index == len(pronunciations) -1:
                    following_c = ''
                elif pronunciation_index == len(pronunciations) -2:
                    following_c = pronunciations[pronunciation_index+1]
                else:
                    following_c = pronunciations[pronunciation_index+1]+pronunciations[pronunciation_index+2]
                VCC_string = last_vowel + following_c
    return VCC_string

def VCC_dict_construction(data):
    substring_dictionary = {}
    count = 0
    for line in data:
        pronunciationString = line.split(':')[1]
        pronunciationString_no_int = ''.join([i for i in pronunciationString if not i.isdigit()])
        pronunciations_no_int = pronunciationString_no_int.split(' ')
        VCC_string = VCC_search(pronunciations_no_int)
        count +=1
        if VCC_string not in substring_dictionary.keys():
            substring_dictionary[VCC_string] = 1
        else:
            substring_dictionary[VCC_string] += 1

    for key in substring_dictionary.keys():
        number = substring_dictionary[key]
        substring_dictionary[key] = float(number )/ count
 
    return substring_dictionary 

def vowel_count_freq(data):
    position_dict = {1:0, 2:0, 3:0, 4:0}
    vowel = ['AA','AE','AH', 'AO', 'AW', 'AY', 'EH', 'ER','EY', 'IH','IY','OW','OY','UH', 'UW']
    total_numberber = 0
    
    for line in data:
        vowel_counter = 0
        total_numberber += 1
        pronunciationString = line.split(':')[1]
        pronunciationString_no_int = ''.join([i for i in pronunciationString if not i.isdigit()])
        pronunciations_no_int = pronunciationString_no_int.split(' ')
        for j in pronunciations_no_int:
            if j in vowel:
                vowel_counter += 1
        if vowel_counter in position_dict.keys():
            position_dict[vowel_counter] += 1
        else:
            position_dict[vowel_counter] = 1
    for i in position_dict.keys():
        position_dict[i] = position_dict[i]/total_numberber 
    return position_dict

def consnt_count_freq(data):
    position_dict = {}
    consnt= {'P','B','CH','D','DH','F','G','HH','JH','K','L','M','N','NG','R','S','SH','T','TH','V','W','Y','Z','ZH'}
    total_numberber = 0
    
    for line in data:
        consnt_counter = 0
        total_numberber += 1
        pronunciationString = line.split(':')[1]
        pronunciationString_no_int = ''.join([i for i in pronunciationString if not i.isdigit()])
        pronunciations_no_int = pronunciationString_no_int.split(' ')
        for j in pronunciations_no_int:
            if j in consnt:
                consnt_counter += 1
        if consnt_counter in position_dict.keys():
            position_dict[consnt_counter] += 1
        else:
            position_dict[consnt_counter] = 1
    for i in position_dict.keys():
        position_dict[i] = position_dict[i]/total_numberber 
    return position_dict

def word_length_freq(data):
    length_dict = {}
    total_numberber = 0
    for line in data:
        total_numberber += 1
        wordString = line.split(':')[0]
        l = len(wordString)
        if l in length_dict.keys():
           length_dict[l] += 1
        else:
            length_dict[l] = 1
    for i in length_dict.keys():
        length_dict[i] = length_dict[i]/total_numberber
    return length_dict

def suffix_freq(data):
    suffix_dict = {}
    suffix = {'ABLE', 'ACY','AL','ANCE','ADE','AGE','AR','AL','ARD','ANT','ARY','ATE','ATIVE','AUX','CHE','CE',
        'CIDE','CRACY','CRAT','CULE','CHIN','DOX','DOM','DATO','DIAN','ER','ED','EE','EN','EER','EMIA','ENCE','ERT','EL',
        'EST','ESQUE','ETIC','ETTE','FUL','FUS','FY','GRAM','GRAMY','HOOD','KIE','IAL','IAN','ILE',
        'IASIS','IBLE','IC','ICAL','IFY','ILY','INE','ING','ION' ,'IOUS','OUS','ISE',
        'ISH','ISM','IST','ITE','ITIS','ITY','IVE','IZATION','IZE','LESS','LET','LIKE',
        'LING','LONGER','LOGIST','LOG','LY','LO','LLOW','MAN','MENT','NA','NESS','NEY','NE','NO','ON','OID','OLOGY',
        'OMA','ONYM','OPIA','OPSY','OR','ORY','OSIS','OSTOMY','OUS','SIR','PATH','PATHY','PHILE','PHONE',
        'PHYTE','PLEGIA','PLEGIC','PNEA','SCOPY','SCOPE','SCRIBE','SCRIPT','SECT','SHIP','SION','SOME',
        'SOPHY','SOPHIC','SKI','SSUS','TH','TION','TOME','TOMY','TROPHY','TUDE','TON','TY','ULAR','UOUS','URE','US','WARD','WORTH',
        'WARE','WISE', 'WARDS', 'Y','None'}
    suffix_dict['None'] = 0
    total_numberber = 0
    for line in data:
        total_numberber += 1
        wordString = line.split(':')[0]
        for i in suffix:
            if i in wordString and i[-1] == wordString[-1]:
               if i in suffix_dict.keys() :
                   suffix_dict[i] += 1
               else:
                   suffix_dict[i] = 1
               break
        else:
            suffix_dict['None'] += 1
    for i in suffix_dict.keys():
        suffix_dict[i] = suffix_dict[i]/total_numberber
                   
    return suffix_dict
                
        
    
def prefix_freq(data):
    prefix_dict = {}
    prefix = {'ANTE', 'ANTI', 'CIRCUM', 'CO', 'DE', 'DIS','EM', 'EN', 'EPI','EX','EXTRA',
          'FORE', 'HYPER', 'IL', 'IM', 'IN', 'IR','INFRA', 'INTER', 'INTRA',
          'MACRO', 'MID', 'MIS', 'MONO', 'NON', 'OMNI', 'POST', 'PRE', 'RE', 'SEMI', 'SUB',
          'SUPER', 'THERM', 'TRANS', 'TRI', 'UN', 'UNI','COM', 'CON', 'CONTRA', 'DE', 'EX',
          'A', 'HOMO', 'MAGN', 'PARA', 'SUB', 'TRANS', 'TRI','None'} 
    prefix_dict['None'] = 0
    total_numberber = 0
    for line in data:
        total_numberber += 1
        wordString = line.split(':')[0]
        for i in prefix:
            if i in wordString and i[0] == wordString[0]:
               if i in prefix_dict.keys():
                   prefix_dict[i] += 1
               else:
                   prefix_dict[i] = 1
            break
        else:
            prefix_dict['None'] += 1
    for i in prefix_dict.keys():
        prefix_dict[i] = prefix_dict[i]/total_numberber
                   
    return prefix_dict

def stressLabel(pronunciations):
    position = 0
    vowelSet = set(['AA','AE','AH', 'AO', 'AW', 'AY', 'EH', 'ER','EY', 'IH','IY','OW','OY','UH', 'UW'])
    for element in pronunciations:
        if (element[:-1] in vowelSet):
            position = position + 1
            if element[-1:] == '1':
                return position
    return position

def word_length_freq(data):
    length_dict = {}
    total_numberber = 0
    for line in data:
        total_numberber += 1
        wordString = line.split(':')[0]
        l = len(wordString)
        if l in length_dict.keys():
           length_dict[l] += 1
        else:
            length_dict[l] = 1
    for i in length_dict.keys():
        length_dict[i] = length_dict[i]/total_numberber
    return length_dict

def isVowel(pronunciation):
    vowel = ['AA','AE','AH', 'AO', 'AW', 'AY', 'EH', 'ER','EY', 'IH','IY','OW','OY','UH', 'UW']
    if pronunciation in vowel:
        return 1
    return 0

#caculate priorprob as feature #update
def probDistribution(lines):
    #Vowel#  AA, AE, AH, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW
    vowel = { 'AA' : 0, 'AE': 0,'AH': 0, 'AO' : 0, 'AW' : 0, 'AY' : 0, 'EH' : 0, 'ER' : 0, 'EY' :0 , 'IH' : 0, 'IY':0, 'OW' : 0, 'OY' : 0, 'UH' :0, 'UW':0
    }
    vBackup = vowel.copy()
    consnt= {'P':0, 'B':0, 'CH':0, 'D':0, 'DH':0, 'F':0, 'G':0, 'HH':0, 'JH':0, 'K':0, 'L':0, 'M':0, 'N':0, 'NG':0, 'R':0, 'S':0, 'SH':0, 'T':0, 'TH':0, 'V':0, 'W':0, 'Y':0, 'Z':0, 'ZH':0}
    consnt['None'] = 0
    consnt['isVowel'] = 0
    consntBackup = consnt.copy()

    for line in lines:
        pronunciationString = line.split(':')[1]
        pronunciations = pronunciationString.split(' ')
        consnt_before_v = 'None'
        last_pronunciation = 'None'
        first_mark = 1
        for pronunciation in pronunciations:
            if pronunciation[-1].isdigit():
                vowelKey = pronunciation[0:-1]
                vBackup[vowelKey] += 1
                if first_mark == 1:
                    consntBackup['None'] += 1
                    first_mark += 1           
                if isVowel(last_pronunciation) == 1:
                    consnt_before_v = 'isVowel'
                    consntBackup['isVowel'] += 1

                if pronunciation[-1] == '1':
                    vowel[vowelKey] += 1
                    consnt[consnt_before_v] += 1
                last_pronunciation = vowelKey
            else:
                consnt_before_v = pronunciation
                consntBackup[pronunciation] = consntBackup[pronunciation] + 1
                last_pronunciation = pronunciation

    vowelProb = dict()
    for vowelKey in vowel.keys():
        rate = float(vowel[vowelKey])/vBackup[vowelKey]
        vowelProb[vowelKey] = rate

    consntProb =dict()
    for consntKey in consnt.keys():
        consntProb[consntKey] = float(consnt[consntKey])/consntBackup[consntKey]
        

    return (vowelProb,consntProb)

#select features 
def featureExtraction(data):
    instance_with_label = []
    
    prefix_list2 = ['CO', 'DE','EM', 'EN', 'EX','IL', 'IM', 'IN', 'IR','RE','UN','DE', 'EX']
    prefix_list3 = ['DIS','EPI','MID', 'MIS','NON','PRE', 'SUB','TRI', 'UNI','COM', 'CON','SUB', 'TRI']
    prefix_list4 = ['ANTE', 'ANTI', 'CIRCUM','EXTRA','FORE', 'HYPER', 'INFRA', 'INTER', 'INTRA',
                    'MACRO', 'MONO', 'OMNI', 'POST','SUPER', 'THERM', 'TRANS', 'CONTRA','HOMO', 'MAGN', 'PARA','TRANS']
    suffix_list2 = ['AL','AR','AL','ER','ED','EE','EN','CE','EL','IC','FY','LY','LO','NE','NO','ON','OR','NA','TH','TY','US']
    suffix_list3 = ['ADE','AGE','ACY','ARD','ANT','ARY','ATE','AUX','CHE','DOX','DOM','EER','ERT','FUL',
               'FUS','EST','KIE','IAL','IAN','ILE','IFY','ILY','INE','ING','ION','OUS','ISE','ISH',
               'ISM','IST','ITE','ITY','IVE','LOG','IZE','LET','MAN','NEY','OID','OUS','SIR','OMA','ORY','TON']
    suffix_list4 = ['ABLE','ANCE','ATIVE','CIDE','CRACY','CRAT','CULE','CHIN','DATO','DIAN','EMIA','ENCE',
               'ESQUE','ETIC','ETTE','GRAM','GRAMY','HOOD','IASIS','IBLE','ICAL','IOUS','ITIS','IZATION','LESS','LIKE',
               'LING','LONGER','LOGIST','LLOW','MENT','NESS','OLOGY','ONYM','OPIA','OPSY','OSIS','OSTOMY','PATH','PATHY','PHILE','PHONE',
               'PHYTE','PLEGIA','PLEGIC','PNEA','SCOPY','SCOPE','SCRIBE','SCRIPT','SECT','SHIP','SION','SOME',
               'SOPHY','SOPHIC','SKI','SSUS','TION','TOME','TOMY','TROPHY','TUDE','ULAR','UOUS','URE','WARD','WORTH',
               'WARE','WISE', 'WARDS']
    vowed_plus_consonant = {'AA':1, 'AE':2, 'AH':3, 'AO':4, 'AW':5, 'AY':6, 'EH':7, 'ER':8, 
                            'EY':9, 'IH':10, 'IY':11, 'OW':12, 'OY':13, 'UH':14, 'UW':15,
                            'P':16, 'B':17, 'CH':18, 'D':19, 'DH':20, 'F':21, 'G':22,'HH':23, 'JH':24, 'K':25, 'L':26, 
                            'M':27, 'N':28, 'NG':29, 'R':30,'S':31, 'SH':32, 'T':33, 'TH':34, 'V':35, 'W':36, 'Y':37, 
                            'Z':38, 'ZH':39}

    pDictionary = prob_proun_number(data)
    substring_dictionary= substring_dict(data)

    consnt_position_freq = consnt_count_freq(data)
    word_length_dict = word_length_freq(data)
    position_1_index = 0; 
    position_3_index = 0;
    
    VC_tail_dictionary = VC_Dict_Construct(data)
    VCC_tail_dictionary = VCC_dict_construction(data)
    vowel_position_freq = vowel_count_freq(data)
    
    for key in vowed_plus_consonant.keys():
        vowed_plus_consonant[key] = vowed_plus_consonant[key]/39
    for line in data:
        word = line.split(':')[0]
        pronunciationString = line.split(':')[1]
        pronunciations_int = pronunciationString.split(' ')
        pronunciationString_no_int = ''.join([i for i in pronunciationString if not i.isdigit()])
        
        pronunciations_no_int = pronunciationString_no_int.split(' ')
        number = count_vowel_number(pronunciations_no_int)
        
        probConsnt = pDictionary[number][1]
        probVowel = pDictionary[number][0]
        vowel_count = 0
        consnt_count = 0
        instance =[]
        consnt_before_v = 'None'
        previous_pronunciation = 'None'
        for pronunciation in pronunciations_int:
            if pronunciation[-1].isdigit():
                vowel_count += 1
                vowelKey = pronunciation[:-1]
                if isVowel(previous_pronunciation )==1:                      
                    consnt_before_v = 'isVowel' 
                instance.append(probConsnt[consnt_before_v])
                instance.append(probVowel[vowelKey])                 
                previous_pronunciation = vowelKey                                        
            else:
                consnt_before_v = pronunciation 
                previous_pronunciation = pronunciation
                consnt_count += 1
        while len(instance) <8:
            instance.append(0)
            
        last_pronoun = vowed_plus_consonant[pronunciations_no_int[-1]]
        #if vowel_count in vowel_position_freq.keys():
        #    instance.append(vowel_position_freq[vowel_count])
        #else:
        #    instance.append(0)
        instance.append(vowel_count)
        instance.append(consnt_count)
        instance.append(last_pronoun)
        
        substrings = partial_string_division(pronunciations_int)
        phrases = []
        for sub in substrings:
            phrases.append(substring_dictionary[sub])

        VC_string = VC_search(pronunciations_no_int)
        instance.append(VC_tail_dictionary[VC_string])

        VCC_string = VCC_search(pronunciations_no_int)
        instance.append(VCC_tail_dictionary[VCC_string])

        prefix = 1
        for i in prefix_list2:
            if i in word:
                prefix = 2
                break
        for i in prefix_list3:
            if i in word:
                prefix = 3
                break
        for i in prefix_list4:
            if i in word:
                prefix = 4
                break
                
        suffix = 1
        for i in suffix_list2:
            if i in word:
                suffix = 2
                break
        for i in suffix_list3:
            if i in word:
                suffix = 3
                break
        for i in suffix_list4:
            if i in word:
                suffix = 4
                break
        
        instance_with_label.append(({'CV_P1':phrases[0],'CV_P2':phrases[1],'CV_P3':phrases[2],'CV_P4':phrases[3],
                        'const_P1':instance[0],'vow_P1':instance[1],'const_P2':instance[2],'vow_P2':instance[3],
                        'const_P3':instance[4],'vow_P3':instance[5],'const_P4':instance[6],'vow_P4':instance[7],
                        'Vowel_count':instance[8],'Constant_count':instance[9],'last_pronoun':instance[10],
                        'VC_tail':instance[11],'VCC_tail':instance[12],
                        'length':len(word),'prefix':prefix,'suffix':suffix},stressLabel(pronunciations_int)))            
    return instance_with_label, pDictionary,substring_dictionary,VC_tail_dictionary,VCC_tail_dictionary, vowel_position_freq,consnt_position_freq, word_length_dict 

def train(data, classifier_file):# do not change the heading of the function
    pickleDump = []
    instance_with_label,pDictionary,substring_dictionary,VC_tail_dictionary,VCC_tail_dictionary, vowel_position_freq,consnt_position_freq, word_length_dict = featureExtraction(data) 
    #train, test = train_test_split(instance_with_label, test_size = 0.2)
    X_train, y_train = list(zip(*instance_with_label))
    encoder = LabelEncoder()
    vectorizer = DictVectorizer(dtype=float, sparse=True)
    X_train = vectorizer.fit_transform(X_train)
    tree = DecisionTreeClassifier(criterion = "gini",max_depth=13)
    tree.fit(X_train, y_train)
    
    #X_test, y_test = list(zip(*test))
    #X_test = vectorizer.fit_transform(X_test)
    #y_pred_class = tree.predict(X_test)

    #f1_result = f1_score(y_test, y_pred_class, average='macro')
    
    dumpFile = open(classifier_file, 'wb')
    pickleDump.append(tree)
    pickleDump.append(pDictionary)
    pickleDump.append(VC_tail_dictionary)
    pickleDump.append(VCC_tail_dictionary)
    pickleDump.append(substring_dictionary)
    pickleDump.append(vowel_position_freq)
    pickleDump.append(consnt_position_freq)
    pickleDump.append(word_length_dict)
    
    
    pickle.dump(pickleDump, dumpFile)
    dumpFile.close()

################# testing #################

#split substring 
def partial_string_division_test(word):
    pronunciations_no_int = word.copy()
    substrings = []
    consnt_before_v = ' '
    previous_pronunciation = ' '
    for pronunciation_index in range(len(pronunciations_no_int)):
        if isVowel(pronunciations_no_int[pronunciation_index]) ==1:
            vowel_current = pronunciations_no_int[pronunciation_index ]
            if isVowel(previous_pronunciation )==1:
                consnt_before_v  = ' '

            substr =consnt_before_v + vowel_current 
            substrings.append(substr)            
            previous_pronunciation = vowel_current   
        else:
            consnt_before_v  = pronunciations_no_int[pronunciation_index]
            previous_pronunciation = pronunciations_no_int[pronunciation_index]
    for i in range(len(substrings)):
        substrings[i]= ''.join([j for j in substrings[i] if j!= ' '])

    while len(substrings)<4:
        substrings.append(' ')
    return substrings


#select features 
def testInstanceFeatureExtraction(data, pDictionary,VC_tail_dictionary,VCC_tail_dictionary,substring_dictionary, vowel_position_freq, consnt_position_freq, word_length_dict):
    instance_with_label = []
    vowed_plus_consonant = {'AA':1, 'AE':2, 'AH':3, 'AO':4, 'AW':5, 'AY':6, 'EH':7, 'ER':8, 
                            'EY':9, 'IH':10, 'IY':11, 'OW':12, 'OY':13, 'UH':14, 'UW':15,
                            'P':16, 'B':17, 'CH':18, 'D':19, 'DH':20, 'F':21, 'G':22,'HH':23, 'JH':24, 'K':25, 'L':26, 
                            'M':27, 'N':28, 'NG':29, 'R':30,'S':31, 'SH':32, 'T':33, 'TH':34, 'V':35, 'W':36, 'Y':37, 
                            'Z':38, 'ZH':39}
    prefix_list2 = ['CO', 'DE','EM', 'EN', 'EX','IL', 'IM', 'IN', 'IR','RE','UN','DE', 'EX']
    prefix_list3 = ['DIS','EPI','MID', 'MIS','NON','PRE', 'SUB','TRI', 'UNI','COM', 'CON','SUB', 'TRI']
    prefix_list4 = ['ANTE', 'ANTI', 'CIRCUM','EXTRA','FORE', 'HYPER', 'INFRA', 'INTER', 'INTRA',
                    'MACRO', 'MONO', 'OMNI', 'POST','SUPER', 'THERM', 'TRANS', 'CONTRA','HOMO', 'MAGN', 'PARA','TRANS']
    suffix_list2 = ['AL','AR','AL','ER','ED','EE','EN','CE','EL','IC','FY','LY','LO','NE','NO','ON','OR','NA','TH','TY','US']
    suffix_list3 = ['ADE','AGE','ACY','ARD','ANT','ARY','ATE','AUX','CHE','DOX','DOM','EER','ERT','FUL',
               'FUS','EST','KIE','IAL','IAN','ILE','IFY','ILY','INE','ING','ION','OUS','ISE','ISH',
               'ISM','IST','ITE','ITY','IVE','LOG','IZE','LET','MAN','NEY','OID','OUS','SIR','OMA','ORY','TON']
    suffix_list4 = ['ABLE','ANCE','ATIVE','CIDE','CRACY','CRAT','CULE','CHIN','DATO','DIAN','EMIA','ENCE',
               'ESQUE','ETIC','ETTE','GRAM','GRAMY','HOOD','IASIS','IBLE','ICAL','IOUS','ITIS','IZATION','LESS','LIKE',
               'LING','LONGER','LOGIST','LLOW','MENT','NESS','OLOGY','ONYM','OPIA','OPSY','OSIS','OSTOMY','PATH','PATHY','PHILE','PHONE',
               'PHYTE','PLEGIA','PLEGIC','PNEA','SCOPY','SCOPE','SCRIBE','SCRIPT','SECT','SHIP','SION','SOME',
               'SOPHY','SOPHIC','SKI','SSUS','TION','TOME','TOMY','TROPHY','TUDE','ULAR','UOUS','URE','WARD','WORTH',
               'WARE','WISE', 'WARDS']
    for key in vowed_plus_consonant.keys():
        vowed_plus_consonant[key] = vowed_plus_consonant[key]/39
    for line in data:
        word = line.split(':')[0]
        pronunciationString = line.split(':')[1]
        pronunciationString_no_int = ''.join([i for i in pronunciationString if not i.isdigit()])
        pronunciations_no_int = pronunciationString_no_int.split(' ')
        number = count_vowel_number(pronunciations_no_int)
        probConsnt = pDictionary[number][1]
        probVowel = pDictionary[number][0]
        vowel_count = 0
        consnt_count = 0
        instance =[]
        consnt_before_v = 'None'
        previous_pronunciation = 'None'
        for pronunciation in pronunciations_no_int:
            if isVowel(pronunciation):
                vowel_count += 1
                vowelKey = pronunciation
                if isVowel(previous_pronunciation )==1:                      
                    consnt_before_v = 'isVowel' 
                instance.append(probConsnt[consnt_before_v])
                instance.append(probVowel[vowelKey])                 
                previous_pronunciation = vowelKey                                        
            else:
                consnt_before_v = pronunciation 
                previous_pronunciation = pronunciation
                consnt_count += 1
        while len(instance) <8:
            instance.append(0)
            
        last_pronoun = vowed_plus_consonant[pronunciations_no_int[-1]]             
        #if vowel_count in vowel_position_freq.keys():
        #    instance.append(vowel_position_freq[vowel_count])
        #else:
        #    instance.append(0)
        instance.append(vowel_count)
        instance.append(consnt_count)
        instance.append(last_pronoun)
        
        substrings = partial_string_division_test(pronunciations_no_int)
        phrases = []
        for sub in substrings:
            if(sub in substring_dictionary.keys()):
                phrases.append(substring_dictionary[sub])
            else:
                phrases.append(0)

        VCC_string = VCC_search(pronunciations_no_int)
        if VCC_string in VCC_tail_dictionary.keys():
            instance.append(VCC_tail_dictionary[VCC_string])
        else:
            instance.append(0)
            
        VC_string = VC_search(pronunciations_no_int)
        if VC_string in VC_tail_dictionary.keys():
            instance.append(VC_tail_dictionary[VC_string])
        else:
            instance.append(0)
                
        prefix = 1
        for i in prefix_list2:
            if i in word:
                prefix = 2
                break
        for i in prefix_list3:
            if i in word:
                prefix = 3
                break
        for i in prefix_list4:
            if i in word:
                prefix = 4
                break
                
        suffix = 1
        for i in suffix_list2:
            if i in word:
                suffix = 2
                break
        for i in suffix_list3:
            if i in word:
                suffix = 3
                break
        for i in suffix_list4:
            if i in word:
                suffix = 4
                break
                
        instance_with_label.append({'CV_P1':phrases[0],
                         'CV_P2':phrases[1],'CV_P3':phrases[2],'CV_P4':phrases[3],'const_P1':instance[0],'vow_P1':instance[1],'const_P2':instance[2],'vow_P2':instance[3],
                        'const_P3':instance[4],'vow_P3':instance[5],'const_P4':instance[6],'vow_P4':instance[7],
                        'Vowel_count':instance[8],'Constant_count':instance[9],'last_pronoun':instance[10],'VC_tail':instance[11],'VCC_tail':instance[12],'length':len(word),'prefix':prefix,'suffix':suffix})
    return instance_with_label



def test(data, classifier_file):# do not change the heading of the function
    prediction = []
    pkl_file = open(classifier_file, 'rb')
    pickleDump = pickle.load(pkl_file)

    testFeatures = testInstanceFeatureExtraction(data, pickleDump[1], pickleDump[2],pickleDump[3],pickleDump[4], pickleDump[5],pickleDump[6],pickleDump[7])
    tree = pickleDump[0]
    vectorizer = DictVectorizer(dtype=float, sparse=True)
    testFeatures = vectorizer.fit_transform(testFeatures)
    prediction = tree.predict(testFeatures)
    return list(map(int,prediction))
