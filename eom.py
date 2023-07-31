import pytesseract
import numpy as np
import PyPDF2
import os
import fitz
from multiprocessing import Pool
from PIL import Image
from multiprocessing import Pool
from nltk.corpus import stopwords
import tabula
import pandas as pd
import torch

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

from utilities import *

class Eom():
    def __init__(self,cfg):
        self.cfg = cfg
        self.pdffile = None
        self.file_name = None
        #Initialize the model
        self.model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-docvqa-large")
        self.processor = Pix2StructProcessor.from_pretrained("google/pix2struct-docvqa-large")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def __call__(self, file, query:str): 
        self.pdffile = file.name
        self.file_name = os.path.basename(self.pdffile)
        # print(self.pdffile)
        # print(self.file_name)
        final_ocr_df = self.ocr_pdf()
        # print(final_ocr_df)
        displayPages,answerPage = self.QnA(query, final_ocr_df)
        # print (displayPages)
        # print (answerPage)
        displayAns = []
        
        imgs = []
        for img_path in displayPages:
            imgs.append(Image.open(img_path))

        # print(imgs)
        #pass page to model to evaluate
        inputs = self.processor(images=imgs, text=[query for _ in range(len(imgs))], return_tensors="pt").to(self.device)
        predictions = self.model.generate(**inputs)
        answers = self.processor.batch_decode(predictions, skip_special_tokens=True)
        
        for i in range(len(displayPages)):
            displayAns.append('Answer page# : '+ str(answerPage[i]) + ', Answer : ' + answers[i])


        return displayAns,displayPages
    
    def ocr_df_using_pytesseract(self, image):
      pytesseract.pytesseract.tesseract_cmd =  self.cfg.tesseract.path
      ocr_df= pytesseract.image_to_data(image, output_type='data.frame')
      ocr_df= ocr_df.dropna().reset_index(drop=True)
      ocr_df =ocr_df.replace(r'^\s*$', np.nan, regex=True)
      ocr_df = ocr_df.dropna().reset_index(drop=True)
      return ocr_df
    
    def pdf_page_ocr(self, page_num):
        saveImagePath = self.cfg.path.saveImage
        output = self.file_name + str(page_num+1) + ".png"
        image= Image.open(os.path.join(os.path.join(saveImagePath, self.file_name), output))
        ocr_df= self.ocr_df_using_pytesseract(image)
        ocr_df['page_num'] = page_num+1
        return ocr_df


    def ocr_pdf(self):
        saveImagePath = self.cfg.path.saveImage
        totalPages = 0
        with open(self.pdffile, 'rb') as file:
            pdfReader = PyPDF2.PdfReader(file)
            totalPages = len(pdfReader.pages) 

        try:
            print("Creating the Directory")
            os.makedirs(os.path.join(saveImagePath, self.file_name))
        except:
            print("Directory already exists")
        else:
            for page_num in range(totalPages):
                doc=fitz.open(self.pdffile)
                page = doc.load_page(page_num)
                pix= page.get_pixmap(dpi=300)
                output = self.file_name + str(page_num+1) + ".png"
                pix.save(os.path.join(os.path.join(saveImagePath, self.file_name), output))
        
        processPool = Pool()
        df_list = processPool.map(self.pdf_page_ocr,range(totalPages))
        processPool.close()
        processPool.join()
        final_ocr_df = pd.concat(df_list,axis = 0)
        return final_ocr_df
    
    def QnA(self, query, final_ocr_df):
        #Get Keywords
        stop_words = set(stopwords.words('english'))
        words= query.split(' ')
        keywords=[]
        for r in words: 
            if r not in stop_words: 
                keywords.append(r)
        keywords = [elem.lower() if isinstance(elem, str) else elem for elem in keywords]

        #Dataframe manipulation
        final_ocr_df['text']= final_ocr_df['text'].astype(str)
        #using fuzzy logic to cover keywords
        print(keywords)
        final_ocr_df['cleaned_text'] = final_ocr_df['text'].apply(filter_and_replace, args = ([keywords]))
        #filtering the row with keywords
        filtered_df = final_ocr_df[final_ocr_df['cleaned_text'].str.lower().isin(keywords)] 
        #part2 
        df= filtered_df.groupby('page_num')['cleaned_text'].unique().apply(lambda x: ', '.join(x)).reset_index()
        df['cleaned_text'] = df['cleaned_text'].str.replace(r'\[|\]|\“|\”', '')
        # split text column into keywords
        df['keywords'] = df['cleaned_text'].str.split(',\s*')
        # apply fuzzy matching to keywords column
        df['keywords'] = df['keywords'].apply(lambda x: list(filter(lambda kw: fuzzy_match(kw, x), x)))
        # remove duplicates from keywords column
        df['keywords'] = df['keywords'].apply(lambda x: list(set(x)))
        df['filtered_keywords'] = df['keywords'].apply(remove_similar_keywords)
        df['filtered_keywords'] = df['filtered_keywords'].apply(len)
        # find maximum value in filtered_keywords column
        max_value = df['filtered_keywords'].max()
        # filter DataFrame to only include rows with maximum value
        max_rows = df[df['filtered_keywords'] == max_value]
        pages_to_be_searched= max_rows['page_num'].tolist()

        keyword_len_list=[]
        dfs=[]
        for page_num in pages_to_be_searched:
            df_list = tabula.read_pdf(self.pdffile, pages = page_num)

            keyword_len=[]
            df_tmp=[]
            for df in df_list:
                data= df.to_numpy().flatten()
                columns=df.columns.to_numpy()
                for i in columns:
                    data= np.append(data,i)
                
                data= data.astype(str)

                data= lowercase_except_digits_floats(data)

                keywords_present = [term for term in keywords for item in data if term in item.lower()]
                unique_elements = set(keywords_present)
                keyword_len.append(len(unique_elements))
                df_tmp.append(df)
            keyword_len_list.append(keyword_len)
            dfs.append(df_tmp)
        
        max_value = None
        max_indexes = []

        for i, sub_lst in enumerate(keyword_len_list):
            if not max_value or max(sub_lst) > max_value:
                max_value = max(sub_lst)
                max_indexes = [(i, sub_lst.index(max_value))]
            elif max(sub_lst) == max_value:
                max_indexes.append((i, sub_lst.index(max_value)))

        displayPages=[]
        answer_page=[]
        for i in max_indexes:
            answer_page.append(pages_to_be_searched[i[0]])
            saveImagePath = self.cfg.path.saveImage
            output = self.file_name + str(pages_to_be_searched[i[0]]) + ".png"
            display_page = (os.path.join(saveImagePath,self.file_name,output))
            displayPages.append(display_page)

        return displayPages,answer_page