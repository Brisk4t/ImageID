# USAGE
# python checkebeddings.py --embedding output/embed.pickle --tolerance 0.1 --unique output/unique.txt
# --output output/similarities.txt


import numpy as np
import argparse
import pickle
import os
from scipy import spatial
import openpyxl


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embedding", required=False,
	help="path to embeddings.pickle file")
ap.add_argument("-x", "--xlsx", required=False,
	help="path to XLSX file")
ap.add_argument("-r", "--range", required=True,
	help="path to XLSX file")
ap.add_argument("-t", "--tolerance", required=True,
	help="tolerance of image similarity")
ap.add_argument("-o", "--output", required=False,
	help="path to store similar image info")
ap.add_argument("-u", "--unique", required=False,
	help="path to store unique image info")
args = vars(ap.parse_args())


def import_from_workbook(col, row):
	return (sheet[col+str(row)].value)

def write_to_workbook(value, col, row):
                sheet[col+str(row+1)] = value

dict_urls = {}

if(args["embedding"]):
        dbfile = open(args["embedding"], 'rb')
        embedding = pickle.load(dbfile)


elif(args["xlsx"]):
        path = args["xlsx"]
        wb = openpyxl.load_workbook(path)
        sheet = wb.active
        embedding = []

        for vector in range(1, int(args["range"])):
                getstring = (import_from_workbook("B", vector).strip('[]'))
                getstring = " ".join(getstring.split())
                dict_urls[import_from_workbook("A", vector)] = np.fromstring(getstring, sep=' ')

        embedding = list(dict_urls.keys())

print(embedding)
print(len(embedding))
tolerance = float(args["tolerance"])
similarity= {}
unique = []


# distance = spatial.distance.cosine(embedding['testimages3\(3).jpg'], embedding['testimages3\(7).jpg'] )



for i in range(0, len(embedding)-1):
        conflicts = []
        similar_url = ""
        for j in range(i+1, len(embedding)):
                distance = spatial.distance.cosine(dict_urls[embedding[i]], dict_urls[embedding[j]])
                if(distance<tolerance):
                        if(args["embedding"]):
                                conflicts.append(embedding[j])
                        elif(args["xlsx"]):
                                similar_url += '; ' + embedding[j]
        
        write_to_workbook(similar_url, 'C', i)                       

        if(args["embedding"]):                        
                if(len(conflicts) != 0):
                        similarity[i] = conflicts
                else:
                        unique.append(i)
                        

wb.save(path)

# with open(args["unique"], mode='wt', encoding='utf-8') as myfile:
#     myfile.write('\n'.join(str(u) for u in unique))

# f = open(args["output"], "w")
# for k in similarity.keys():
#     f.write("{}:\n{}\n\n".format(k, similarity[k]))

#____________________________________________________________________________________________
# for i in similarity:
#         conflicts = []
#         query_val = embedding[i]
#         for value in embed_values:
        #         if(abs(query_val-value)<tolerance):
        #                 conflicts.append(embed_values.index(value)+1) 
        # similarity[i] = conflicts


# f = open("log.txt", "w")
# f.write("{\n")
# for k in similarity.keys():
#     f.write("'{}':'{}'\n".format(k, similarity[k]))
# f.write("}")
# f.close()