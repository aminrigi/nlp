from transformers import pipeline
sentiment_classifier = pipeline('sentiment-analysis')


def get_sentiment_bert(text):
    text = str(text)
    text = text.lower()
    result = sentiment_classifier(text)
    if result[0]['label'] == 'NEGATIVE':
        sentiment = -1*result[0]['score']
    else:
        sentiment = result[0]['score']
        
    return(sentiment)

def get_sentiments(df, textCol):
    """
    This function gets a df, the name of the column that has text (corpus).
    It then adds a column named sentiment. With sentiment scores. If sentiment
    is less than zero, it's negative; otherwise positive.
    """
    df['sentiment'] = df[textCol].apply(get_sentiment_bert)
    
    return df
    