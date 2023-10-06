# Data Anaytics Course Project (In Progress 09.25.2023)
## Project Title: The Impact Artificial Intelligence has on Future Jobs

Dataset Source: Kaggle - https://www.kaggle.com/datasets/manavgupta92/from-data-entry-to-ceo-the-ai-job-threat-index
Apps: RapidMiner and Python
Created By: Group 4

### Objective:
Class was assigned to find a dataset and practice data analysis techniques to preprocess, transform, and output a data analysis tool. Group 4 decided to work of a dataset that contains records of AI Impact on varying job titles and the team decided to create a tool that will help users see the job titles that have been heavily impact by AI. This will assist users determine where they should be focusing their efforts in their carrer.

### Procedures:
#### 1. Preprocessing
Much of the dataset has been cleaned by the original owner of the dataset from Kaggle. However, there is actually concerningly little amount of data to analyze effectively. As there are many varying job titles, it would make it difficult to analyze these titles effectively. The group has shown interest in using clustering algorithms so we make groups of similar jobs, instead of thousands of individual jobs. However this enlightened another problem. Many of these titles are short (no bigger than 5 words), many titles share similar responsibilities, and many titles share similar titles words but hold different responsibilities. For example:
  - Manager vs Director of Human Resources
  - Cook vs Line Cook
  - Chief Data Officer vs Chief Executive Officer

This has lead us to looking into solutions in clustering like Short Text Cluster and solutions to add more data by web scraping commonly known job description.
