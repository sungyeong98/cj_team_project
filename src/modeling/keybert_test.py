from keybert import KeyBERT
import pandas as pd
import os

script_dir = os.path.dirname(__file__)
relative_path1 = "../../data/processed/Hannanum_dataset.csv"
file_path1 = os.path.join(script_dir, relative_path1)

script_dir = os.path.dirname(__file__)
relative_path2 = "../../data/processed/Okt_dataset.csv"
file_path2 = os.path.join(script_dir, relative_path2)

df_han=pd.read_csv(file_path1)
df_okt=pd.read_csv(file_path2)

model=KeyBERT()

df_han['ex_words']=''
df_okt['ex_words']=''

for idx,row in df_han.iterrows():
    n=len(row['imp_words'].split(','))
    if n>len(row['hannanum_nouns'].split(',')):
        n=len(row['hannanum_nouns'].split(','))
    keywords1=model.extract_keywords(row['hannanum_nouns'],top_n=n)
    df_han.at[idx,'ex_words']=','.join([word for word,_ in keywords1])
for idx,row in df_okt.iterrows():
    n=len(row['imp_words'].split(','))
    if n>len(row['okt_nouns'].split(',')):
        n=len(row['okt_nouns'].split(','))
    keywords2=model.extract_keywords(row['okt_nouns'],top_n=n)
    df_okt.at[idx,'ex_words']=','.join([word for word,_ in keywords2])

save_path1='../../data/result/hannanum_test_dataset.csv'
save_path2='../../data/result/okt_test_dataset.csv'
save_file_path1=os.path.join(script_dir,save_path1)
save_file_path2=os.path.join(script_dir,save_path2)

df_han.to_csv(save_file_path1)
df_okt.to_csv(save_file_path2)