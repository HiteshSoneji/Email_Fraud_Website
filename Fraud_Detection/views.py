#Importing the required libraries
from django.shortcuts import render
import email
import imaplib
import pandas as pd
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import gensim
from nltk import word_tokenize
import re
import string
from nltk.corpus import stopwords
import numpy as np
import copy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import time
import timeit
from sklearn import metrics

#Index will be the starting page of the website, where the user logins
def index(request):

    #Accessing the user email and password
    if request.method == 'POST':

        email_id = request.POST.get('emailid', False)
        password = request.POST.get('password', False)

        print(email_id, password)

        #Login in into the user's account
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        (retcode, capabilities) = mail.login(email_id, password)
        mail.list()
        mail.select('inbox')

        #Reading the unread emails
        n = 0
        (retcode, messages) = mail.search(None, '(UNSEEN)')
        if retcode == 'OK':
            for num in messages[0].split():
                print('Processing ')
                n = n + 1
                typ, data = mail.fetch(num, '(RFC822)')
                for response_part in data:
                    if isinstance(response_part, tuple):
                        original = email.message_from_bytes(response_part[1])
                        # print (original['From'])
                        # print (original['Subject'])
                        raw_email = data[0][1]
                        raw_email_string = raw_email.decode('utf-8')
                        email_message = email.message_from_string(raw_email_string)
                        for part in email_message.walk():
                            if (part.get_content_type() == "text/plain"):  # ignore attachments/html
                                body = part.get_payload(decode=True)
                                save_string = str(r"D:\Final Year Project\Model\Test_email\email" + str(n) + ".txt")
                                myfile = open(save_string, 'a', errors='ignore')
                                myfile.write('From:\n')
                                myfile.write(original['From'] + '\n')
                                myfile.write('Subject:\n')
                                myfile.write(original['Subject'] + '\n')
                                myfile.write('Body:\n')
                                myfile.write(body.decode('utf-8', errors='ignore'))
                                myfile.write('****\n')
                                myfile.close()
                            else:
                                continue

                        typ, data = mail.store(num, '+FLAGS', '\\Seen')
                if n==10:
                    break

        print(n)

        #Saving it into a temporary csv file
        df = pd.DataFrame()
        path = 'D:\Final Year Project\Model\Test_email'

        listing = os.listdir(path)

        #Separating the body and subject of the email
        body = []
        subject =[]
        for i in range(len(listing)):
            filepath = 'D:\Final Year Project\Model\Test_email\email'+str(i+1)+'.txt'
            with open(filepath, 'r', encoding="utf-8", errors='ignore') as file:
                data = file.read()
            data = data.replace(r'\r', '')
            data = data.replace(r'\n', '')
            d = data.split('Body:')
            body.append(d[1])
            d1 = data.index("Subject:") + len("Subject:")
            d2 = data.index("Body:", d1)
            d = data[d1:d2]
            subject.append(d)
            print(subject)
            os.remove(filepath)

        df['Body'] = body
        df['Subject'] = subject

        print(df)

        df.to_csv('D:\Final Year Project\Model\Testing_Emails.csv')

        df = pd.read_csv('D:\Final Year Project\Model\Testing_Emails.csv')
        df.dropna(subset=['Body'], inplace=True)
        # df.drop_duplicates(subset=[, 'To', 'Date', 'Received', 'Subject', 'Body'], keep='first')
        df.reset_index(inplace=True)

        data1 = df.filter(['Body'], axis = 1)

        #Cleaning the data
        def remove_punct(text):
            text_nopunct = ''
            text_nopunct = re.sub('[' + string.punctuation + ']', '', str(text))
            return text_nopunct

        data1['Body'] = data1['Body'].apply(lambda x: remove_punct(x))
        print(data1['Body'])


        data1.Body.astype('string')

        tokens = [word_tokenize(sen) for sen in data1.Body]

        def lower_token(tokens):
            return [w.lower() for w in tokens]

        lower_tokens = [lower_token(token) for token in tokens]
        print(lower_tokens)

        stoplist = stopwords.words('english')

        def removeStopWords(tokens):
            return [word for word in tokens if word not in stoplist]

        filtered_words = [removeStopWords(sen) for sen in lower_tokens]
        data1['Body_final'] = [' '.join(sen) for sen in filtered_words]
        data1['Body_tokens'] = filtered_words

        all_training_words = [word for tokens in data1["Body_tokens"] for word in tokens]
        training_sentence_lengths = [len(tokens) for tokens in data1["Body_tokens"]]
        TRAINING_VOCAB = sorted(list(set(all_training_words)))
        print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
        print("Max sentence length is %s" % max(training_sentence_lengths))

        word2vec = gensim.models.KeyedVectors.load_word2vec_format('D:\Final Year Project\Model\GoogleNews-vectors-negative300.bin', binary=True)
        print(word2vec.vector_size)

        def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
            if len(tokens_list) < 1:
                return np.zeros(k)
            if generate_missing:
                vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
            else:
                vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
            length = len(vectorized)
            summed = np.sum(vectorized, axis=0)
            averaged = np.divide(summed, length)
            return averaged

        #Finding word embeddings

        def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
            embeddings = clean_comments['Body_tokens'].apply(
                lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing))
            print(embeddings)
            return list(embeddings)

        training_embeddings = get_word2vec_embeddings(word2vec, data1, generate_missing=True)


        data1['Body_embeddings'] = training_embeddings

        data2 = copy.deepcopy(data1[['Body_final', 'Body_embeddings']])
        print(data2.head())

        MAX_SEQUENCE_LENGTH = 2000
        EMBEDDING_DIM = 300

        # data_train, data_test = train_test_split(data1, test_size=0.7, random_state=42)
        data_train = data1
        print(data_train.size)

        tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
        tokenizer.fit_on_texts(data_train["Body_final"].tolist())
        training_sequences = tokenizer.texts_to_sequences(data_train["Body_final"].tolist())
        train_word_index = tokenizer.word_index
        print("Found %s unique tokens." % len(train_word_index))

        train_cnn_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)


        train_embedding_weights = np.zeros((len(train_word_index) + 1, EMBEDDING_DIM))
        for word, index in train_word_index.items():
            train_embedding_weights[index, :] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
        print(train_embedding_weights.shape)

        #Passing it through the saved model

        from keras.models import load_model

        model1 = load_model('D:\Final Year Project\Model\saved_model')
        # summarize model.
        print(model1.summary())

        labels = [1, 0]

        x_train = train_cnn_data

        predictions = model1.predict(x_train)

        prediction_labels=[]
        for p in predictions:
            prediction_labels.append(labels[np.argmax(p)])

        print(df.Subject)
        print(prediction_labels)
        os.remove('D:\Final Year Project\Model\Testing_Emails.csv')
        no = []
        for i in range(len(subject)):
            no.append(int(i+1))
        data = zip(no, subject, prediction_labels)
        context = {
            'data':data
        }
        return render(request, 'home.html', context)


    return render(request, 'index.html')

#Home will display the results
def home(request):

    return render(request, 'home.html')
