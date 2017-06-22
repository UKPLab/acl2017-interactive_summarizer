import sys, os.path as path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
import argparse
import numpy as np

def aggregate(file_name):
    baselines = []
    with open(file_name, 'r') as fp:
        lines = fp.read().splitlines()
    i = 0
    items = []
    while(i < len(lines)):
        if lines[i].startswith('###'): 
            i += 3
            while(1):
                if lines[i].startswith('###'):
                    print items
                    baselines.append(items)
                    i += 1
                    break
                else:
                    print lines[i]
                    system, R1, R2, SU4 = lines[i].split(' ')
                    print system
                    if system == 'UB1':
                        items = []
                        print 'IAM IN'
                    items.append([float(R1), float(R2), float(SU4)])
                    i += 1
        else:
            i += 1
    
    systems = ['UB1', 'UB2', 'Luhn', 'LexRank', 'TextRank', 'LSA', 'KL']
    for index in range(len(systems)):        
        R1 = np.array([x[index][0] for x in baselines])
        R2 = np.array([x[index][1] for x in baselines])
        SU4 = np.array([x[index][2] for x in baselines])
        print '%s, R1: %4f, R2: %4f SU4: %4f' % (systems[index], np.mean(R1), np.mean(R2), np.mean(SU4))
            
def get_args():
    ''' This function parses and return arguments passed in'''

    parser = argparse.ArgumentParser(description='Baselines Results Aggregator')
    parser.add_argument('-l', '--summary_length', type= str, help='Scores file', required=False)
    parser.add_argument('-d', '--data_set', type= str, help='Year of the data set', required=True)

    args = parser.parse_args()
    summary_len = args.summary_length
    data_set = args.data_set
    return summary_len, data_set

if __name__ == '__main__':
    summary_len, data_set = get_args()
    if summary_len:
        data_path = '../data/scores/baselines_%s_%s' % (data_set, summary_len)
    else:
        data_path = '../data/scores/baselines_%s' % (data_set)
    aggregate = aggregate(data_path)