import pandas as pd
import textblob as tb
from sklearn.metrics import confusion_matrix

df = pd.read_csv("./imdb_labelled copy.txt", sep='\t', names=['sentence','sentiment'], error_bad_lines=False, engine='python')
lexicon_results = []
for i in range(len(df['sentence'])):
    text = tb.TextBlob(df.loc[[i], ['sentence']].to_string())
    if text.polarity > 0:
        text.polarity = 1
    else:
        text.polarity = 0
    lexicon_results.append(text.polarity)

df['lexicon_results'] = lexicon_results
print('Confusion Matrix for Lexicon: \n', confusion_matrix(df['sentiment'], df['lexicon_results']))
print('Actual: \n', df['sentiment'])
print('predicted: \n', df['lexicon_results'])
