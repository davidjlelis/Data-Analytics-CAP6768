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
        for j in range(runs):

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
        job_df.at[i, 'Major Group Code'] = soc_code[0:2]
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
        title1 = job_df['Job titiles'][i]
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
                title2 = col_title_df['Title'][j]

                # print('Comparing ', title1, ' and ', title2)
                score = calculate_similarity(title1=title1.lower(), title2=title2.lower())
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
                score = calculate_similarity(title1=title1.lower(), title2=title2.lower())
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
            job_df.at[i, 'Major Group Code'] = soc_code[0:2]
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

    final_df.to_excel('output/preprocessing/Standardized Job Titles and SOC Codes'+file_name+'.xlsx', index=False)

    return final_df

def kmeans_cluster(data, attribute,testing):
    if testing == True:
        data = data[data['Job Description'].notna()]
         # print('Kmeans clustering skipped')
         # return

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

    results.to_excel('output/clusters/Clusters_'+attribute+'_kmeans.csv', index=False)

def gsdmm_cluster(data, attribute, testing):
    if testing == True:
        data = data[data['Job Description'].notna()]
         # return
    
     # -- Start of Clustering using GSDMM based on Job Titles
    sentence = data[attribute]
    text = [text.split() for text in sentence]
    #print (text)
    V = compute_V(texts=text)
    mgp = MovieGroupProcess(K=50, alpha = 0.1, beta = 0.1, n_iters = 100)
    
    gsdmm_results = pd.DataFrame()
    gsdmm_results['job_titles'] = sentence
    gsdmm_results['cluster']  = mgp.fit(sentence, V)

    gsdmm_results.to_excel('output/clusters/Clusters_'+attribute+'_gsdmm.csv', index=False)

    # -- End of Clustering using GSDMM based on Job Titles

def output_full_dataset(job_ai_standard_df, occupation_df, major_code_df, testing):
    if testing == True:
         print('Outputting full dataset skipped')
         return

    # Get Job Description merging on SOC code
    full_df = job_ai_standard_df.merge(occupation_df, left_on='SOC Code', right_on='O*NET-SOC Code', how='left')
    full_df = full_df.merge(major_code_df, left_on='Major Group Code', right_on = 'Major Group Code', how='left')



    output = full_df[[
        'Job titiles',
        'AI Impact',
        'Tasks',
        'AI models',
        'AI_Workload_Ratio',
        'Domain',
        'Standard Job Title',
        'O*NET-SOC Code',
        'Title',
        'Major Group Code',
        'SOC or O*NET-SOC 2019 Title',
        'Description',
        'Match Score',
        'Matched by'
    ]]

    output.rename(columns={
        'Job titiles': 'Job Title',
        'AI Impact': 'AI Impact',
        'Tasks': 'Tasks',
        'AI models': 'AI Models',
        'AI_Workload_Ratio': 'AI Workload Ratio',
        'Domain': 'Domain',
        'Standard Job Title': 'O*Net Job Title',
        'O*NET-SOC Code': 'O*Net SOC Code',
        'Title': 'O*Net SOC Title',
        'Major Group Code': 'O*Net Major Group Code',
        'SOC or O*NET-SOC 2019 Title': 'O*Net Major Group Title',
        'Description': 'Job Description',
        'Match Score': 'Match Score',
        'Matched by': 'Matched By'
    }, inplace=True)

    output.to_excel('output/preprocessing/Full Dataset_v2.xlsx', index=False)

    return full_df

def get_input():
    result = ''

    result = input('What are we doing today? (Enter 1, 2,3 or 4)\n1. Test the full program.\n2. Run the preprocessing programs.\n3. Run the clustering algorithm programs.\n4. Run the full program.\n\nInput: ')
    
    if result in [1, 2, 3]:
        return result
    elif result == 4:
        result = input('Running the full program may take hours to run due to running through tens of thoundsand of data.\nIt is recommended to download the preloaded data to run. Are you sure you want to run the whole program? (Enter Y or N)\n\nInput: ')
        if result == 'Y':
            return result
        else:
            result = get_input()

    return result

def preprocess(testing):
    file = ('from-data-entry-to-ceo-the-ai-job-threat-index//My_Data.csv')
    occupation_data_file = 'db_28_0_excel/Occupation Data.xlsx'
    alternate_titles_file = 'db_28_0_excel/Alternate Titles.xlsx'
    soc_structure_file = 'db_28_0_excel/SOC_Structure.xlsx'

    job_ai_df = pd.read_csv(file)
    occupation_df = pd.read_excel(occupation_data_file)
    alt_title_df = pd.read_excel(alternate_titles_file)
    soc_structure_df = pd.read_excel(soc_structure_file)

    major_group_df = soc_structure_df[['Major Group', 'SOC or O*NET-SOC 2019 Title']]
    major_group_df = major_group_df.dropna().reset_index()

    major_group_df['Major Group Code'] = np.nan

    for i in range(len(major_group_df)):
        major_group_df.at[i, 'Major Group Code'] = major_group_df['Major Group'][i][0:2]

    # -- Start of extracting job description data from O*NET
    
    size = round(len(job_ai_df)/9)-1

    job_ai_df['Standard Job Title'] = np.nan
    job_ai_df['SOC Code'] = np.nan
    job_ai_df['Major Group Code'] = np.nan
    job_ai_df['Match Score'] = np.nan
    job_ai_df['Matched by'] = np.nan

    job_ai_standard_df = pd.DataFrame()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(get_job_title_and_soc_code, args=[job_ai_df[size*i:size*(i+1)], alt_title_df, '_'+str(i+1), testing]) for i in range(9)]

        for f in concurrent.futures.as_completed(results):
            job_ai_standard_df = job_ai_standard_df.append(f.result(), ignore_index=True)
            print(f.result())

    job_ai_standard_df.to_excel('output/preprocessing/Standardized Job Titles and SOC Codes.xlsx', index=False)
    output_full_dataset(job_ai_standard_df=job_ai_standard_df, occupation_df=occupation_df, major_code_df=major_group_df, testing=False)
    
    # pass
    
    return

def cluster(testing):
    file = 'output/preprocessing/Full Dataset.xlsx'
    full_df = pd.read_excel(file)

    # Cluster based on Raw Job Title
    kmeans_cluster(data=full_df, attribute='job_titiles', testing=testing)
    gsdmm_cluster(data=full_df, attribute='job_titiles', testing=testing)

    # Cluster based on job description
    kmeans_cluster(data=full_df, attribute='Job Description', testing=testing)
    gsdmm_cluster(data=full_df, attribute='Job Description', testing=testing)
    # pass
    return

def main():
    user_input = int(get_input())

    if user_input == 1:
        preprocess(testing=True)
        cluster(testing=True)
    elif user_input == 2:
        preprocess(testing=False)
    elif user_input == 3:
        cluster(testing=False)
    elif user_input == 4:
        preprocess(testing=False)
        cluster(testing=False)

    

    # -- End of extracting job description data from O*NET

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