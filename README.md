## Naive Bayesian Classifier of spam mails
### The code implements a simple letter classification system based on a Bayesian approach. 

### Main description

The code implements a simple letter classification system based on a Bayesian approach. The main steps include:\
-Preprocessing of text (removing numbers, reducing to lowercase, lemmatization)\
-Training of the model by calculating the frequencies of words in two categories ("spam" and "ham")\
-Classification of new letters based on calculated probabilities\
This code can serve as the basis for more complex text analysis systems

### List of used libraries

- math, os.path, re
- pandss
- nltk

### Installation and launch

- Move the nbc_of_spam.py to the folder with your project
- Install the necessary modules and libraries

### Library installation code and an example of using the program

1) You need to install the "nltk" and "pandas"
```terminal
pip install nltk
pip install pandas
```
2) After that, you need to install the necessary "nltk" modules
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```
3) Using the classifier
```python
from Task_SPAM import *
training_set = train_machine("testdata", "testdata_ready")
check_mail('Hello!!!', training_set) 
# The check_mail function returns the evaluation result ('SPAM' or 'HAM')
```

### Mecessary links
https://www.kaggle.com/datasets/abdallahwagih/spam-emails --- testdata download link








