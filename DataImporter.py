import glob
import pandas as pd
import os 
from pathlib import Path
from pathlib import Path

class DataImporter:

    def __init__(self):
        self.data = None
        self.dataframe = None
        self.content = None
        
    def makeDf(self):    
        files = glob.glob('**/*.txt',recursive=True)
      
        
        try: files.pop('topics.txt')
        except: pass
        try: files.pop('keywords.txt')
        except: pass
        #files = glob.glob( path + folderName + "/*.txt")
        fileDocs = []
        fileDictContainer = []
        for file in files:
            if 'topics.txt' in file or  'keywords.txt' in file or 'topics2.txt' in file:
                continue
          
            with open(file, 'r') as fd:
                contents = fd.read().replace('\n', '')
                fileDocs.append(contents)
                fileDocsDict = {}
                try:
                    fileDocsDict['fileName'] = file.split('/')[-1]
                    fileDocsDict['fileFolder'] = file.split('/')[-2]
                    fileDocsDict['fileContent'] = contents
                except:
                    continue
                fileDictContainer.append(fileDocsDict)
        return fileDictContainer
    

    def parseContentFiles(self):
        fileDictContainer = []
        
        folder_dict_Container = self.makeDf()
        fileDictContainer.extend(folder_dict_Container)
        df = pd.DataFrame.from_dict(fileDictContainer)
        self.contents = df['fileContent'].tolist()
        return df
    
    def getContent(self):
        return self.contents
    
