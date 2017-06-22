import sys, os.path as path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import argparse

import matplotlib.pyplot as plt
import os
from summarizer.utils.reader import read_csv

class Plot(object):
    def load_data(self, filename, cluster_id, info_per_user):
        self.cluster = cluster_id
        self.rows = read_csv(filename)
        self.info_num = info_per_user

    def plot_scores(self, rouge_type='R1_score'):
        ub_scores = self.rows[0]
        users = len(ub_scores[1:])/self.info_num

        y = [[] for user in range(users)]
        for iteration in range(0,len(self.rows)):
            row = self.rows[iteration]
            for user in range(users):
                if rouge_type == 'R1_score':
                    val = row[1+user*self.info_num]
                    if val != "":
                        y[user].append(float(val))
                if rouge_type == 'R2_score':
                    val = row[2+user*self.info_num]
                    if val != "":
                        y[user].append(float(val))


        plt.subplot(211)
        plt.plot(range(len(y[0][1:])), len(y[0][1:]) *[y[0][0]], 'k--', label='UB1')
        plt.plot(range(len(y[0][1:])), y[0][1:], 'r', label='User 1')
        plt.plot(range(len(y[1][1:])), len(y[1][1:]) *[y[1][0]], 'k--', label='UB2')
        plt.plot(range(len(y[1][1:])), y[1][1:], 'b', label='User 2')

        plt.legend(loc="lower right")

        plt.subplot(212)
        plt.plot(range(len(y[2][1:])), len(y[2][1:]) *[y[2][0]], 'k--', label='UB3')
        plt.plot(range(len(y[2][1:])), y[2][1:],'y', label='User 3')
        plt.plot(range(len(y[3][1:])), len(y[3][1:]) *[y[3][0]], 'k--', label='UB4')
        plt.plot(range(len(y[3][1:])), y[3][1:], 'g', label='User 4')

        plt.legend(loc="lower right")
        plt.xlabel("No. of iterations")
        plt.ylabel(rouge_type)
        plt.yscale("linear")
        plt.show()

def get_args():
    ''' This function parses and return arguments passed in'''

    parser = argparse.ArgumentParser(description='Results Aggregator')
    parser.add_argument('-w', '--writer_output', type= str, help='Writer Output Directory', required=True)

    args = parser.parse_args()
    woutput = args.writer_output
    return woutput

if __name__ == '__main__':

    woutput = get_args()

    plotter = Plot()
    data_path = '../data/%s' % (woutput)
    score_type = 'R2_score'

    
    for fileid in os.listdir(data_path):
        filename = '%s/%s' % (data_path, fileid)
        cluster_id = fileid[:-3]
        plotter.load_data(filename, cluster_id, 5)
        plotter.plot_scores(score_type)

        
    
