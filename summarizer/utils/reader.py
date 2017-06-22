import csv, codecs
   
def read_csv(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar= '"')
        rows = [row for row in reader]
        return rows

def read_file(filename):
    with codecs.open(filename, 'r', 'utf-8', errors='ignore') as fp:
        return fp.read()