import re
import math
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os.path


lemma_dict = WordNetLemmatizer() #---Lemmatizer for word normalization
stop_words_dict = set(stopwords.words('english')).union({'.', ','})


def transform_mail(mail):   ###---Text processing: tokenization, lemmatization, stopword filtering.
    mail_text = re.sub(r"(^|\s+)(\d+([\.,]\d+)?)(?=\s|$)", "", mail).lower()
    tokens = word_tokenize(mail_text)
    lemma_tokens = [lemma_dict.lemmatize(word) for word in tokens if word not in stop_words_dict]
    return lemma_tokens


def sort_mail(mail, category, table):   ###---Sorting words by category and updates the counters.
    for i in mail:
        if category == 'ham':
            if i in table:
                table[i][1] += 1
            else:
                table[i] = [1, 1, 0, 0]
        elif category == 'spam':
            if i in table:
                table[i][0] += 1
            else:
                table[i] = [1, 1, 0, 0]
    return table


def count_total_words(table):   ###---Calculating the total number of words and probabilities for each word
    total_spam = 0
    total_ham = 0
    for key in table.keys():
        total_spam += table[key][0]
        total_ham += table[key][1]
    return [total_spam, total_ham]


def calculate_probability(mail, table, total_spam, total_ham, help_parameter=False):    ###---Determining the probability of being spam
    probability_spam = math.log(1 / (1 + total_ham/total_spam))
    probability_ham = math.log(1 / (1 + total_spam/total_ham))
    k = 0
    if help_parameter: #---Parameter can help in case of uneven distribution of the number of words in classes.
        k = 1
    for i in mail:
        if i not in table.keys():
            probability_ham += math.log(1 / (total_ham + total_spam*k))
            probability_spam += math.log(1 / (total_spam + total_ham*k))
        elif i in table:
            probability_ham += math.log(table[i][3])
            probability_spam += math.log(table[i][2])
    return [probability_spam, probability_ham]


def train_machine(name_raw_data, name_ready_file): ###---Training the classifier
    if os.path.exists(str(name_ready_file)+'.csv'): #---Checking for a ready-made file with trained data
        csv_df = pd.read_csv(str(name_ready_file)+'.csv', index_col=0)
        first_column_name = csv_df.index.name
        csv_dict = csv_df.to_dict(orient='index') #---Uploading data from a CSV file
        table_words = {key: list(value.values()) for key, value in csv_dict.items()} #---Transforming the dictionary into a convenient form
        total = [int(num) for num in re.findall(r'\d+', first_column_name)] #---Extracting the amount of spam and non-spam
    else: #---If the file is missing - training the model
        table_words = {}
        data = pd.read_csv(str(name_raw_data)+'.csv')
        for row in data.itertuples(index=True): #---Going through each letter
            table_words = sort_mail(transform_mail(str(row.Message)), str(row.Category), table_words) #---Sorting words into categories
        total = count_total_words(table_words) #---Counting the amount of spam and non-spam
        for key in table_words.keys(): #---Calculate probabilities for each word
            table_words[key][2] = table_words[key][0] / total[0]
            table_words[key][3] = table_words[key][1] / total[1]
        df_orient = pd.DataFrame.from_dict(table_words, orient='index', columns=['in_s', 'in_h', 'prob_s', 'prob_h']) #---Forming the Data Frame
        df_orient.index.name = '%s{s-%d, h-%d}' % ('word', total[0], total[1]) #---Saving information about the number of spam and non-spam messages in index name
        df_orient.to_csv(str(name_ready_file)+'.csv', index=True) #---Saving the results to a CSV file
    return [table_words, total] #---Forming training set for function check_mail


def check_mail(mail, training_set, help_parameter=False):
    table = training_set[0]
    total = training_set[1]
    check_probability = calculate_probability(transform_mail(mail), table, total[0], total[1], help_parameter)
    verdict = 'HAM'
    if check_probability[0] >= check_probability[1]:
        verdict = 'SPAM'
    return [verdict, check_probability]
