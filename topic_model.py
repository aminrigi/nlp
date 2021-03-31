from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn

# topic modeling 
def get_lda_vis(corpus, n_topics, tfidf = False, learn_approach = 'batch'):
    """
    Returns an interactive visualisation for lda models
    
    
    tfidf: should the tfidf normalisation applied? my research shows no for LDA, maybe yes for NMF. Hence it's False by default
    learn_approach: online or batch, the former is faster, but less accurate
    """
    
    if tfidf == False:
        tf_vectorizer = CountVectorizer(analyzer='word',       
                                 min_df=2,                       
                                 stop_words='english',             
                                 lowercase=True,                   
                                 token_pattern='[a-zA-Z0-9]{3,}',  
                                 max_features=3000,          
                                )

        dtm_tf = tf_vectorizer.fit_transform(corpus)
        lda = LatentDirichletAllocation(n_components=n_topics, learning_method=learn_approach, n_jobs = -1 ) # -1 uses all CPUs
        lda.fit(dtm_tf)
        feature_names = tf_vectorizer.get_feature_names()
        viz = pyLDAvis.sklearn.prepare(lda, dtm_tf, tf_vectorizer)
    else:
        tfidf_vectorizer = TfidfVectorizer(analyzer='word',       
                                 min_df=2,                       
                                 stop_words='english',             
                                 lowercase=True,                   
                                 token_pattern='[a-zA-Z0-9]{3,}',  
                                 max_features=3000,          
                                )
        dtm_tfidf = tfidf_vectorizer.fit_transform(corpus)
        lda = LatentDirichletAllocation(n_components=n_topics, learning_method=learn_approach, n_jobs = -1)
        lda.fit(dtm_tfidf)
        feature_names = tfidf_vectorizer.get_feature_names()
        viz = pyLDAvis.sklearn.prepare(lda, dtm_tfidf, tfidf_vectorizer)

    # Print the top 10 words per topic
    n_words = 15
    topic_list = []
    for topic_idx, topic in enumerate(lda.components_):
        top_n = [feature_names[i]
                 for i in topic.argsort()
                 [-n_words:]][::-1]
        top_features = ' '.join(top_n)
        topic_list.append(f"topic_{'_'.join(top_n[:3])}") 

        print(f"Topic {topic_idx}: {top_features}")

    return(viz)





#getting the optimal number of toics

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

def optimal_topic_no(corpus, topic_range):
    """
    Find the optimal number of topics within a range for a corpus
    """

    # Options to try with our LDA
    # Beware it will try *all* of the combinations, so it'll take ages
    search_params = {
      'n_components': topic_range,
      'learning_decay': [.7]#, .5]
    }
    
    tf_vectorizer = CountVectorizer(analyzer='word',       
                                 min_df=2,                       
                                 stop_words='english',             
                                 lowercase=True,                   
                                 token_pattern='[a-zA-Z0-9]{3,}',  
                                 max_features=2000,          
                                )
    dtm_tf = tf_vectorizer.fit_transform(corpus)
    
    # Set up LDA with the options we'll keep static
    model = LatentDirichletAllocation(learning_method='batch')

    # Try all of the options
    gridsearch = GridSearchCV(model, param_grid=search_params, n_jobs=5, verbose=1)
    gridsearch.fit(dtm_tf)

    # What did we find?
    print("Best Model's Params: ", gridsearch.best_params_)
    print("Best Log Likelihood Score: ", gridsearch.best_score_)
    
    return gridsearch.best_params_


########## Printing topics #################

def display(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print("\nTopic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            print(documents[doc_index])
            
            
            
def display_topics(corpus, n_topics, no_top_words=10, no_top_documents=4):
    """
    corpus: is documents
    n_topics: number of topics
    """
    tf_vectorizer = CountVectorizer(analyzer='word',       
                                 min_df=2,                       
                                 stop_words='english',             
                                 lowercase=True,                   
                                 token_pattern='[a-zA-Z0-9]{3,}',  
                                 max_features=3000,          
                                )

    dtm_tf = tf_vectorizer.fit_transform(corpus)
    lda = LatentDirichletAllocation(n_components=n_topics, learning_method='batch', n_jobs = -1 ) # -1 uses all CPUs
    lda.fit(dtm_tf)
    feature_names = tf_vectorizer.get_feature_names()
    lda_W = lda.transform(dtm_tf)
    lda_H = lda.components_

    
    print("LDA Topics")
    display(lda_H, lda_W, feature_names, corpus.to_list(), no_top_words, no_top_documents)




