'''
Project Title: The Impact Artificial Intelligence has on Future Jobs
Project step: Preprocessing
Purpose: Preprocessing AI impact on Jobs dataset to a structure so we can effectively mine and analyze the data.

Creators: Group 4
    Marina Lombardo (N01578798)
    Andrea Garcia Fernandez (N01561009)
    David Lelis (N00957151)
    Stephanie El-Bahri (N01404796)


Problem: Dataset doesn't contain enough data to analyze effectively as job titles vary for roles that share very similar 
    responsibilities. Cases may involve the following:
        1. Job titles are very short like Chef, Teacher, Manager, etc.
        2. Job titles that share very similar responsibilies but have different titles like Data Analyst vs Data Scientist
        3. Job titles that share similart titles but have different responsibilities like Chief Data Office vs Chief Executive Officer

Solutions:
    1. Short Text Clustering Algorithms
        - Since the data is limited, finding an algorithm that maximizes the data given would be a good shot to group these roles 
            based on the given data.
    2. Web Scraping Job Descriptions and Clustering
        - Add more data by looking up the well-known responsilibities of these roles and clustering based on this data.
'''

# Import libraries
import pandas as pd
import numpy as np
import opendatasets as od
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from gsdmm.gsdmm import MovieGroupProcess

import difflib as dl
import spacy
import concurrent.futures

# -- use to check for library versions
#print(pd.__version__)
#print(np.__version__)

# -- use to download dataset, if not downloaded. Comment out if downloaded.
# od.download("https://www.kaggle.com/datasets/manavgupta92/from-data-entry-to-ceo-the-ai-job-threat-index")

nlp = spacy.load("en_core_web_md")

def compute_V(texts):
        V = set()
        for text in texts:
            for word in text:
                V.add(word)
        return len(V)

def calculate_similarity(title1, title2):
    doc1 = nlp(title1)
    doc2 = nlp(title2)

    return doc1.similarity(doc2)

def run_SequenceMatcher(job_df, onet_title_df, file_name, testing):

    if testing == True:
         #print('Skipping Standardizing Job Titles')
         runs = 3
         #return
    else: runs = len(job_df)

    for i in range(runs):
        title1 = job_df['Job titiles'][i]
        standard_job_title = ''
        soc_code = ''
        score = 0
        highest_score = 0
        matcher = 'SequenceMatcher'

        print('********************** SequenceMatcher Process'+file_name+' **************************',
              '\nPosition ', i, ' out of ', len(job_df), 
              '\nFinding Standard Title for', title1,
              '\n********************** SequenceMatcher Process'+file_name+' **************************')


        # Step 1. Run SequenceMatcher
        for j in range(len(onet_title_df)):

            title2 = onet_title_df['Title'][j]
            # Use SequenceMatcher to compare title1 and title2
            score = dl.SequenceMatcher(None, title1.lower(), title2.lower()).ratio()

            # If found perfect match, assign and break loop
            if score == 1:
                highest_score = score
                soc_code = onet_title_df['O*NET-SOC Code'][j]
                standard_job_title = title2
                break
            # If score is greater than the current highest score, assign and repeat
            elif score > highest_score:
                highest_score = score
                soc_code = onet_title_df['O*NET-SOC Code'][j]
                standard_job_title = title2

        # Loop through alternate titles for SequenceMatcher
        for j in range(len(onet_title_df)):
            title2 = onet_title_df['Alternate Title'][j]

            # If already found perfect match, break loop to assign
            if highest_score == 1:
                break

            score = dl.SequenceMatcher(None, title1.lower(), title2.lower()).ratio()
            # If found perfact match, assign and break loop
            if score == 1:
                highest_score = score
                soc_code = onet_title_df['O*NET-SOC Code'][j]
                standard_job_title = title2
                break
            # if score is greater than the highest score, then assign and repeat loop
            elif score > highest_score:
                highest_score = score
                soc_code = onet_title_df['O*NET-SOC Code'][j]
                standard_job_title = title2

        job_df.at[i, 'Standard Job Title'] = standard_job_title
        job_df.at[i, 'Match Score'] = highest_score
        job_df.at[i, 'SOC Code'] = soc_code
        job_df.at[i, 'Matched by'] = matcher

        print('\n********************** SequenceMatcher Process'+file_name+' **************************',
              '\nStandard Job Title found!',
              '\nJob Title: ', title1,'\nStandard Title: ', standard_job_title,
              '\nScore: ', highest_score,
              '\nSOC Code: ', soc_code,
              '\nMatched by: ', matcher,
              '\n********************** SequnceMatcher Process'+file_name+' **************************')
        
    print('SequenceMatcher Process'+file_name, ' finished!')
    return job_df

def run_NLP(job_df, onet_title_df, file_name, testing):

    if testing == True:
         #print('Skipping Standardizing Job Titles')
         runs = 3
         #return
    else: runs = len(job_df)

    col_title_df = onet_title_df[['Title', 'O*NET-SOC Code']].drop_duplicates()
    col_alt_title_df = onet_title_df[['Alternate Title', 'O*NET-SOC Code']].drop_duplicates()

    col_title_df.reset_index(level=None, drop=False, inplace=True, col_level=0, col_fill='')
    col_alt_title_df.reset_index(level=None, drop=False, inplace=True, col_level=0, col_fill='')

    for i in range(runs):
        title1 = job_df['Job titiles'][i].lower()
        standard_job_title = job_df['Standard Job Title'][i]
        soc_code = job_df['SOC Code'][i]
        score = 0
        highest_score = job_df['Match Score'][i]
        matcher = job_df['Matched by'][i]

        print('********************** NLP Process'+file_name+' **************************',
              '\nPosition ', i, ' out of ', len(job_df), 
              '\nFinding Standard Title for', title1,
              '\n********************** NLP Process'+file_name+' **************************')
    
        if highest_score >= 0.98:
            pass

        # If highest_score is < 0.98, run NLP matcher
        elif highest_score < 0.98:
            matcher = 'NLP'
            # print('Check', search_title_df)

            for j in range(len(col_title_df)):
                title2 = col_title_df['Title'][j].lower()

                # print('Comparing ', title1, ' and ', title2)
                score = calculate_similarity(title1=title1, title2=title2)
                if score < 0.8 or score < highest_score:
                    continue
                elif score == 1:
                    highest_score = score
                    soc_code = col_title_df['O*NET-SOC Code'][j]
                    standard_job_title = title2
                    break
                elif score > highest_score:
                    highest_score = score
                    soc_code = col_title_df['O*NET-SOC Code'][j]
                    standard_job_title = title2

            for j in range(len(col_alt_title_df)):
                title2 = col_alt_title_df['Alternate Title'][j]

                if highest_score == 1:
                    break
                # print('Comparing ', title1, ' and ', title2)
                score = calculate_similarity(title1=title1, title2=title2)
                if score < 0.8 or score < highest_score:
                    continue
                elif score == 1:
                    highest_score = score
                    soc_code = col_alt_title_df['O*NET-SOC Code'][j]
                    standard_job_title = title2
                    break
                elif score > highest_score:
                    highest_score = score
                    soc_code = col_alt_title_df['O*NET-SOC Code'][j]
                    standard_job_title = title2
        # 1b. If the match does not score 0.98, move to step 2 

            job_df.at[i, 'Standard Job Title'] = standard_job_title
            job_df.at[i, 'Match Score'] = highest_score
            job_df.at[i, 'SOC Code'] = soc_code
            job_df.at[i, 'Matched by'] = matcher

        print('\n********************** NLP Process'+file_name+' **************************',
              '\nStandard Job Title found!',
              '\nJob Title: ', title1,'\nStandard Title: ', standard_job_title,
              '\nScore: ', highest_score,
              '\nSOC Code: ', soc_code,
              '\nMatched by: ', matcher,
              '\n********************** NLP Process'+file_name+' **************************')
        
    print('NLP Process'+file_name, ' finished!')

    return job_df

#def get_job_title_and_soc_code(job_df, onet_title_df, file_name, testing):
def get_job_title_and_soc_code(args):
    job_df= args[0]
    onet_title_df = args[1] 
    file_name = args[2]
    testing = args[3]
    '''
    Logic:
    For each title in Job Titles:
        1. Run the SequenceMatcher program to look for matches
            a. If the match scores 0.98 or higher, move to step 3
            b. If the match does not score 0.98, move to step 2
        2. Run the nlp program to look for matches
        3. Add word and repeat with the next word
    '''

    job_df.reset_index(level=None, drop=False, inplace=True, col_level=0, col_fill='')

    SequenceMatcher_df = run_SequenceMatcher(job_df, onet_title_df, file_name, testing)
    final_df = run_NLP(SequenceMatcher_df, onet_title_df, file_name, testing)

    final_df.to_excel('output/Standardized Job Titles and SOC Codes'+file_name+'.xlsx', index=False)

    return final_df

    '''
    # Attempt 2 - using SequenceMatcher and NLP sequentially

    #for i in range(5): # testing loop case
    for i in range(runs):
        title1 = job_df['Job titiles'][i]
        standard_job_title = ''
        soc_code = ''
        score = 0
        highest_score = 0
        matcher = ''

        print('********************** Process'+file_name+' **************************',
              '\nPosition ', i, ' out of ', len(job_df), 
              '\nFinding Standard Title for', title1,
              '\n********************** Process'+file_name+' **************************')


        # Step 1. Run SequenceMatcher
        for j in range(len(onet_title_df)):

            title2 = onet_title_df['Title'][j]
            # Use SequenceMatcher to compare title1 and title2
            score = dl.SequenceMatcher(None, title1.lower(), title2.lower()).ratio()

            # If found perfect match, assign and break loop
            if score == 1:
                highest_score = score
                soc_code = onet_title_df['O*NET-SOC Code'][j]
                standard_job_title = title2
                break
            # If score is greater than the current highest score, assign and repeat
            elif score > highest_score:
                highest_score = score
                soc_code = onet_title_df['O*NET-SOC Code'][j]
                standard_job_title = title2

        # Loop through alternate titles for SequenceMatcher
        for j in range(len(onet_title_df)):
            title2 = onet_title_df['Alternate Title'][j]

            # If already found perfect match, break loop to assign
            if highest_score == 1:
                break

            score = dl.SequenceMatcher(None, title1.lower(), title2.lower()).ratio()
            # If found perfact match, assign and break loop
            if score == 1:
                highest_score = score
                soc_code = onet_title_df['O*NET-SOC Code'][j]
                standard_job_title = title2
                break
            # if score is greater than the highest score, then assign and repeat loop
            elif score > highest_score:
                highest_score = score
                soc_code = onet_title_df['O*NET-SOC Code'][j]
                standard_job_title = title2
        matcher = 'SequenceMatcher'
        
        print('-----------SequenceMatcher Results-------------------',
              '\nJob Title: ', title1,
              '\nStandard Title: ', standard_job_title,
              '\nScore: ', highest_score,
              '\nSOC Code: ', soc_code,
              '\nMatched by: ', matcher,
              '\n---------------------------------------------------')
        
        # 1a. If the match scores 0.98 or higher, skip NLP
        if highest_score >= 0.98:
            pass

        # If highest_score is < 0.98, run NLP matcher
        elif highest_score < 0.98:
            matcher = 'NLP'
            for j in range(len(onet_title_df['Title'])):
                title2 = onet_title_df['Title'][j]

                # print('Comparing ', title1, ' and ', title2)
                score = calculate_similarity(title1=title1, title2=title2)
                if score < 0.8 or score < highest_score:
                    continue
                elif score == 1:
                    highest_score = score
                    soc_code = onet_title_df['O*NET-SOC Code'][j]
                    standard_job_title = title2
                    break
                elif score > highest_score:
                    highest_score = score
                    soc_code = onet_title_df['O*NET-SOC Code'][j]
                    standard_job_title = title2

            for j in range(len(onet_title_df['Alternate Title'])):
                title2 = onet_title_df['Alternate Title'][j]

                if highest_score == 1:
                    break
                # print('Comparing ', title1, ' and ', title2)
                score = calculate_similarity(title1=title1, title2=title2)
                if score < 0.8 or score < highest_score:
                    continue
                elif score == 1:
                    highest_score = score
                    soc_code = onet_title_df['O*NET-SOC Code'][j]
                    standard_job_title = title2
                    break
                elif score > highest_score:
                    highest_score = score
                    soc_code = onet_title_df['O*NET-SOC Code'][j]
                    standard_job_title = title2
        # 1b. If the match does not score 0.98, move to step 2 

        job_df.at[i, 'Standard Job Title'] = standard_job_title
        job_df.at[i, 'Score'] = highest_score
        job_df.at[i, 'SOC Code'] = soc_code
        job_df.at[i, 'Matched by'] = matcher

        print('\n********************** Process'+file_name+' **************************',
              '\nStandard Job Title found!',
              '\nJob Title: ', title1,'\nStandard Title: ', standard_job_title,
              '\nScore: ', highest_score,
              '\nSOC Code: ', soc_code,
              '\nMatched by: ', matcher,
              '\n********************** Process'+file_name+' **************************')

    job_df.to_excel('output/Standardized Job Titles and SOC Codes'+file_name+'.xlsx', index=False)
    return job_df
    '''
    '''
    # Attempt 1 - using diffllib and SequenceMatcher
    #for i in range(5): # testing loop case
    for i in range(len(job_df)):
        standard_job_title = ''
        soc_code = ''
        score = 0
        highest_score = 0

        job_title = job_df['Job titiles'][i].lower()
        
        print('Getting standard title for ', job_title)
        for j in range(len(onet_title_df)):
            onet_title = onet_title_df['Title'][j].lower()
            score = dl.SequenceMatcher(None, job_title, onet_title).ratio()
            if score > highest_score:
                highest_score = score
                soc_code = onet_title_df['O*NET-SOC Code'][j]
                standard_job_title = onet_title_df['Title'][j]

        for j in range(len(onet_title_df)):
            onet_title = onet_title_df['Alternate Title'][j].lower()
            score = dl.SequenceMatcher(None, job_title, onet_title).ratio()
            if score > highest_score:
                highest_score = score
                soc_code = onet_title_df['O*NET-SOC Code'][j]
                standard_job_title = onet_title_df['Alternate Title'][j]
    '''

def kmeans_cluster(data, attribute,testing):
    if testing == True:
         print('Kmeans clustering skipped')
         return

    sentence = data[attribute]
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorized_documents = vectorizer.fit_transform(sentence)
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(vectorized_documents.toarray())
    
    num_clusters= 45
    kmeans = KMeans(n_clusters = num_clusters,
                    n_init = 5,
                    max_iter=500,
                    random_state=42)
    kmeans.fit(vectorized_documents)

    results = pd.DataFrame()
    results['job_titles'] = sentence
    results['cluster'] = kmeans.labels_

    # print(results.sample(5))

    results.to_csv('Clusters_'+attribute+'_kmeans.csv', index=False)

def gsdmm_cluster(data, attribute, testing):
    if testing == True:
         print('GSDMM clustering skipped')
         return
    
     # -- Start of Clustering using GSDMM based on Job Titles
    sentence = data[attribute]
    text = [text.split() for text in sentence]
    #print (text)
    V = compute_V(texts=text)
    mgp = MovieGroupProcess(K=50, alpha = 0.1, beta = 0.1, n_iters = 100)
    
    gsdmm_results = pd.DataFrame()
    gsdmm_results['job_titles'] = sentence
    gsdmm_results['cluster']  = mgp.fit(sentence, V)

    gsdmm_results.to_csv('Clusters_'+attribute+'_gsdmm.csv', index=False)

    # -- End of Clustering using GSDMM based on Job Titles

def output_full_dataset(job_ai_standard_df, occupation_df, testing):
    if testing == True:
         print('Outputting full dataset skipped')
         return

    full_df = job_ai_standard_df.merge(occupation_df, left_on='SOC Code', right_on='O*NET-SOC Code', how='left')

    full_df.to_excel('output/Full Dataset.xlsx', index=False)

    return full_df


def main():
    file = ('from-data-entry-to-ceo-the-ai-job-threat-index//My_Data.csv')
    job_ai_df = pd.read_csv(file)
    '''
    input_testing = input('For sake of time, should we skip all of the following? (Y/N)',
                          '\n1. K-Means Clustering on Raw Job Titles',
                          '\n2. GSDMM Clustering on Raw Job Titles',
                          '\n3. Getting Standardized Job Titles from O*NET'
                          '\n4. Outputting Raw Job Titles and O*NET data to get job descriptions')

    if input_testing == 'Y':
        testing_kmeans = True
        testing_gsdmm = True
        testing_standard_job_titles = True
        testing_output_full_dataset = True
    else:
        input_kmeans = input('Do you want to skip K-Means Clustering on Raw Job Titles? (Y/N)')
        input_gsdmm = input('Do you want to skip GSDMM Clustering on Raw Job Titles? (Y/N)')
        input_standard_job_titles = input('Do you want to skip getting Standardized Job Titles from O*NET? (Y/N)')
        input_output_full_dataset = input('Do you want to skip outputting Raw Job Titles and O*NET data to get job descriptions? (Y/N)')

        if input_kmeans == 'Y':
            testing_kmeans = True
        elif input_kmeans != 'Y':
            testing_kmeans = False

        if input_gsdmm == 'Y':
            testing_gsdmm = True
        elif input_gsdmm != 'Y':
            testing_gsdmm = False

        if input_standard_job_titles == 'Y':
            testing_standard_job_titles = True
        elif input_standard_job_titles != 'Y':
            testing_standard_job_titles = False

        if input_output_full_dataset == 'Y':
            testing_output_full_dataset = True
        elif input_output_full_dataset != 'Y':
            testing_output_full_dataset = False

    # auto-assign test -- comment out when running the full project
    testing_kmeans = True
    testing_gsdmm = True
    testing_standard_job_titles = True
    testing_output_full_dataset = True
    '''
    # print(data.head())

    # Cluster based on Raw Job Title
    #kmeans_cluster(data=job_ai_df, attribute='job_titiles', testing=True)
    #gsdmm_cluster(data=job_ai_df, attribute='job_titiles', testing=True)

    # -- Start of extracting job description data from O*NET
    occupation_data_file = 'db_28_0_excel/Occupation Data.xlsx'
    alternate_titles_file = 'db_28_0_excel/Alternate Titles.xlsx'

    occupation_df = pd.read_excel(occupation_data_file)
    alt_title_df = pd.read_excel(alternate_titles_file)

    # print(job_ai_df[0:1568])
    # print(job_ai_df[1569:3138])
    # print(job_ai_df[3139:4706])

    size = round(len(job_ai_df)/9)-1
    #print(size)
    #print(job_ai_df[0:factor])
    #print(job_ai_df[factor:factor*2])
    #print(job_ai_df[factor*2:])

    job_ai_df['Standard Job Title'] = np.nan
    job_ai_df['SOC Code'] = np.nan
    job_ai_df['Match Score'] = np.nan
    job_ai_df['Matched by'] = np.nan

    #print(alt_title_df['Alternate Title'])
    job_ai_standard_df = pd.DataFrame()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(get_job_title_and_soc_code, args=[job_ai_df[size*i:size*(i+1)], alt_title_df, '_'+str(i+1), True]) for i in range(9)]

        for f in concurrent.futures.as_completed(results):
            job_ai_standard_df = job_ai_standard_df.append(f.result(), ignore_index=True)
            print(f.result())

    job_ai_standard_df.to_excel('output/Standardized Job Titles and SOC Codes.xlsx', index=False)
    full_df = pd.DataFrame()
    full_df = output_full_dataset(job_ai_standard_df=job_ai_standard_df, occupation_df=occupation_df, testing=False)

    # -- End of extracting job description data from O*NET

    # Cluster based on job description
    # kmeans_cluster(data=full_df, attribute='Job Description', testing=False)
    # gsdmm_cluster(data=full_df, attribute='Job Description', testing=False)

    '''
    # visual graph of the clusters

    colors = ['red', 'green', 'blue', 'yellow', 'black']
    cluster = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']

    for i in range(num_clusters):
        plt.scatter(reduced_data[kmeans.labels_ == i, 0],
                    reduced_data[kmeans.labels_ == i, 1],
                    s = 10, color=colors[i],
                    label=f' {cluster[i]}')

    plt.legend()
    plt.show()
    '''

if __name__ == '__main__':
     main()