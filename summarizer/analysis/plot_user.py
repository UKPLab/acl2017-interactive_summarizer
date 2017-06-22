import sys, os.path as path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
import numpy as np
import argparse

import matplotlib.pyplot as plt
import os
from summarizer.utils.reader import read_csv
import matplotlib as mpl
mpl.use('pgf')

def figsize(scale):
    fig_width_pt = 455.24408                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def savefig(filename):
    plt.savefig( "figs/" + '{}.pgf'.format(filename))
    plt.savefig("figs/" + '{}.pdf'.format(filename))


class Plot(object):
    def load_data(self, filename, cluster_id, info_per_user):
        self.cluster = cluster_id
        self.rows = read_csv(filename)
        self.info_num = info_per_user

    def get_scores(self, rouge_type='R1_score'):
        ub_scores = self.rows[0]
        self.users = len(ub_scores[1:])/self.info_num

        y = [[] for user in range(self.users)]
        for iteration in range(0,len(self.rows)):
            row = self.rows[iteration]
            for user in range(self.users):
                if rouge_type == 'R1_score':
                    val = row[1+user*self.info_num]
                    if val != "":
                        y[user].append(float(val))
                if rouge_type == 'R2_score':
                    val = row[2+user*self.info_num]
                    if val != "":
                        y[user].append(float(val))
        return y
    
    def plot_scores(self, labels, scores, filename):
        self.users = 2
        f, axis = plt.subplots(2, sharex=True, sharey=False, figsize=(4, 6))
        colors = ['g','b','r', '#8E4585']
        linestyles = ['->', '-o', '-', '-x']
        iterations= 8

        for i in range(self.users):
            for index, score in enumerate(scores):
                y = score
                if index == 0:
                    axis[i].plot(range(len(y[i][1:])), len(y[i][1:]) *[y[i][0]], 'k--', label = 'Upper bound', linewidth=2)
                #axis[i].plot(range(len(y[i][1:])), len(y[i][1:]) *[y[i][0]], 'k--', label = 'Upper-bound')
                if i>0:
                    axis[i].plot(range(len(y[i][1:])), y[i][1:], linestyles[index], color=colors[index], label='%s' % labels[index], linewidth=2)
                else:
                    axis[i].plot(range(len(y[i][1:])), y[i][1:], linestyles[index], color=colors[index], label='%s' % labels[index], linewidth=2)
            axis[i].set_title('User:%s' % str(i+1))
            axis[i].set_xticks(np.arange(0, iterations, 1))
            axis[i].set_xticklabels(np.arange(0, iterations, 1))
            axis[i].set_ylabel('ROUGE 2', fontsize=13)
            axis[i].grid(True)
        
        plt.legend(loc="best", fontsize=9)
        plt.xlabel("\# Iterations", fontsize=15)
        plt.yscale("linear")
        plt.xlim(0,iterations)
        plt.tight_layout()
        savefig(filename)

def get_args():
    ''' This function parses and return arguments passed in'''

    parser = argparse.ArgumentParser(description='Users Aggregator')
    parser.add_argument('-d', '--data_set', type= str, help='Dataset Name', required=True)
    parser.add_argument('-l', '--summary_len', type= str, help='Summary Size', required=False)
    args = parser.parse_args()
    
    data_set = args.data_set
    size = args.summary_len
    return data_set, size

if __name__ == '__main__':

    data_set, size = get_args()
    methods = ['active_learning2', 'active_learning','ilp_feedback', 'accept_reject']
    labels = ['Active+','Active', 'Joint', 'Accept']
    plotter = Plot()
    
    pgf_with_latex = {                      # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": True,                # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 10,               # LaTeX default is 10pt font.
        "text.fontsize": 10,
        "legend.fontsize": 10,               # Make the legend/label fonts a little smaller
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.figsize": figsize(0.85),     # default fig size of 0.9 textwidth
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
            r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
            ]
        }
    mpl.rcParams.update(pgf_with_latex)
    
    if size!= None:
        data_path = '../data/scores/%s/%s_%s_%s' % (data_set, methods[2], data_set, size)
    else:
        data_path = '../data/scores/%s/%s_%s' % (data_set, methods[2], data_set)
    
    score_type = 'R2_score'
    
    for fileid in os.listdir(data_path):
        scores = []
        cluster_id = fileid[:-3]
        print cluster_id
        for method in methods:
            if size!= None:
                filename = '../data/scores/%s/%s_%s_%s/%s' % (data_set, method, data_set, size, fileid)
            else:
                filename = '../data/scores/%s/%s_%s/%s' % (data_set, method, data_set, fileid)
            plotter.load_data(filename, cluster_id, 6)
            scores.append(plotter.get_scores(score_type))
        filename = "users_%s_%s" % (data_set, fileid)
        plotter.plot_scores(labels, scores, filename)