from __future__ import print_function
import sys, os.path as path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from summarizer.utils.reader import read_csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import os
import argparse
from sets import Set
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

class Aggregator(object):
    def __init__(self):
        self.count_ub_reach = []
        self.no_accepts = []
        self.no_rejects = []
        self.iterations = []
        self.R1_scores = []
        self.R2_scores = []
        self.SU4_scores = []
        self.clusters = []
        self.summaries = []

    def load_data(self, filename, cluster_id, info_per_user):
        self.clusters.append(cluster_id)
        self.rows = read_csv(filename)
        self.info_num = info_per_user

    def plot_distribution(self, mean, sigma, array):
        vlines = [mean-(1*sigma), mean, mean+(1*sigma)]
        for val in vlines:
            plt.axvline(val, color='k', linestyle='--')

        bins = np.linspace(mean-(4*sigma), mean+(4*sigma), 200)
        plt.hist(array, bins, alpha=0.5)

        y = mlab.normpdf(bins, mean, sigma)
        plt.plot(bins, y, 'r--')
        plt.subplots_adjust(left=0.15)
        plt.show()
        print(mean, sigma)

    def get_summaries(self, cluster_id, user_id):
        ub_summary = self.summaries[cluster_id][user_id][0]
        soa_summary = self.summaries[cluster_id][user_id][1]
        last = len(self.R2_scores[cluster_id][user_id])
        print(self.R2_scores[cluster_id][user_id][0], self.R2_scores[cluster_id][user_id][1], self.R2_scores[cluster_id][user_id][last-1]  )
        print('UB summary:\n', ub_summary)
        print('SOA summary:\n', soa_summary)

    def print_min_max_avg_std(self, array, tag):
        np_array = np.array(array)
        mean = np.mean(np_array)
        sigma = np.std(np_array)
        print('%s\nmin:%4f max:%4f avg:%4f std:%4f' % (tag, np.min(np_array), np.max(np_array), mean, sigma))
        '''
        if tag == 'R2 ub_soa diff:' or tag == 'R2 ub_last diff:':
            self.plot_distribution(mean, sigma, array)
        '''
        return mean

    def print_results(self):
        ub_count, total_ub = 0.0, 0.0
        accepts, rejects, iterations = [], [], []
        R1_ub_last_diff, R2_ub_last_diff, SU4_ub_last_diff, other_accepts, other_rejects, other_iterations = [], [], [], [], [], []
        other_R1_ub_last_diff, other_R2_ub_last_diff = [], []
        reach_R1_ub_last_diff, reach_R2_ub_last_diff = [], []
        R1_ub_soa_diff, R2_ub_soa_diff, SU4_ub_soa_diff = [], [], []
        R1_system, R2_system, SU4_system = [], [], []
        R1_UB, R2_UB, SU4_UB = [], [], []
        R1_SOA, R2_SOA, SU4_SOA = [], [], []
        R1_soa_last_diff, R2_soa_last_diff, SU4_soa_last_diff = [], [], []

        #num_clusters = len(self.count_ub_reach)
        num_clusters = len(self.R1_scores)

        for cluster in range(num_clusters):
            no_users = len(self.count_ub_reach[cluster])
            for user in range(no_users):
                total_ub += 1
                last = len(self.R1_scores[cluster][user])

                index = np.argmax(np.array(self.R2_scores[cluster][user][1:last]))

                index = index+1
                '''
                print(self.R2_scores[cluster][user][1:last])
                print(self.R2_scores[cluster][user][1:index+1])
                print(self.no_accepts[cluster][user][1:index+1])
                '''
                accepts.append(sum(self.no_accepts[cluster][user][1:index]))
                rejects.append(sum(self.no_rejects[cluster][user][1:index]))
                iterations.append(index)

                R1_ub_last_diff.append(self.R1_scores[cluster][user][0]-self.R1_scores[cluster][user][index])
                R2_ub_last_diff.append(self.R2_scores[cluster][user][0]-self.R2_scores[cluster][user][index])
                SU4_ub_last_diff.append(self.SU4_scores[cluster][user][0]-self.SU4_scores[cluster][user][index])

                R1_ub_soa_diff.append(self.R1_scores[cluster][user][0]-self.R1_scores[cluster][user][1])
                R2_ub_soa_diff.append(self.R2_scores[cluster][user][0]-self.R2_scores[cluster][user][1])
                SU4_ub_soa_diff.append(self.SU4_scores[cluster][user][0]-self.SU4_scores[cluster][user][1])

                R1_system.append(self.R1_scores[cluster][user][index])
                R2_system.append(self.R2_scores[cluster][user][index])
                SU4_system.append(self.SU4_scores[cluster][user][index])

                R1_UB.append(self.R1_scores[cluster][user][0])
                R2_UB.append(self.R2_scores[cluster][user][0])
                SU4_UB.append(self.SU4_scores[cluster][user][0])

                R1_SOA.append(self.R1_scores[cluster][user][1])
                R2_SOA.append(self.R2_scores[cluster][user][1])
                SU4_SOA.append(self.SU4_scores[cluster][user][1])

                R1_soa_last_diff.append(self.R1_scores[cluster][user][1] - self.R1_scores[cluster][user][index])
                R2_soa_last_diff.append(self.R2_scores[cluster][user][1] - self.R2_scores[cluster][user][index])
                SU4_soa_last_diff.append(self.SU4_scores[cluster][user][1] - self.SU4_scores[cluster][user][index])

                if self.count_ub_reach[cluster][user] == 1:
                    ub_count += 1
                    reach_R1_ub_last_diff.append(self.R1_scores[cluster][user][0]-self.R1_scores[cluster][user][index])
                    reach_R2_ub_last_diff.append(self.R2_scores[cluster][user][0]-self.R2_scores[cluster][user][index])

                if self.count_ub_reach[cluster][user] == 0:
                    other_accepts.append(sum(self.no_accepts[cluster][user][:-1]))
                    other_rejects.append(sum(self.no_rejects[cluster][user][:-1]))
                    other_iterations.append(self.iterations[cluster][user])
                    other_R1_ub_last_diff.append(self.R1_scores[cluster][user][0]-self.R1_scores[cluster][user][index])
                    other_R2_ub_last_diff.append(self.R2_scores[cluster][user][0]-self.R2_scores[cluster][user][index])
            '''
            if self.count_ub_reach[cluster].count(1) == no_users:
                print('All One Cluster: %s' % (self.clusters[cluster]))
                print('Iterations', self.iterations[cluster])
            if self.count_ub_reach[cluster].count(0) == no_users:
                print('All Zero Cluster: %s' % (self.clusters[cluster]))
                print('Iterations', self.iterations[cluster])
            '''

        print('No. of clusters:%d' % (num_clusters))
        print('Total UB_Reach:%d/%d' % (ub_count, total_ub))
        print('Total rejects avg:%d min,max:%d,%d' % (np.mean(np.array(rejects)), min(rejects),  max(rejects)))
        print('Total accepts avg:%d min,max:%d,%d' % (np.mean(np.array(accepts)), min(accepts), max(accepts)))
        print('Total iterations avg:%d min,max:%d,%d' % (np.mean(np.array(iterations)), min(iterations), max(iterations)))


        max_diff_index = R2_soa_last_diff.index(min(R2_soa_last_diff))
        max_cluster, max_user = max_diff_index/self.users, max_diff_index%self.users
        #print('Cluster with max difference:', self.clusters[max_cluster], max_user)
        #self.get_summaries(max_cluster, max_user)

        '''
        self.print_min_max_avg_std(reach_R1_ub_last_diff, 'Reached R1 ub_last diff:\n')
        self.print_min_max_avg_std(reach_R2_ub_last_diff, 'Reached R2 ub_last diff:\n')
        self.print_min_max_avg_std(other_R1_ub_last_diff, 'Other R1 ub_last diff:\n')
        self.print_min_max_avg_std(other_R2_ub_last_diff, 'Other R2 ub_last diff:\n')
        '''
        '''
        self.print_min_max_avg_std(R1_ub_soa_diff, 'R1 ub_soa diff:')
        self.print_min_max_avg_std(R2_ub_soa_diff, 'R2 ub_soa diff:')
        self.print_min_max_avg_std(R1_ub_last_diff, 'R1 ub_last diff:')
        self.print_min_max_avg_std(R2_ub_last_diff, 'R2 ub_last diff:')
        '''

        avg_r1_ub = self.print_min_max_avg_std(R1_UB, 'R1 UB')
        avg_r2_ub = self.print_min_max_avg_std(R2_UB, 'R2 UB')
        avg_r4_ub = self.print_min_max_avg_std(SU4_UB, 'SU4 UB')

        avg_r1_sys = self.print_min_max_avg_std(R1_system, 'R1 system')
        avg_r2_sys = self.print_min_max_avg_std(R2_system, 'R2 system')
        avg_r4_ub = self.print_min_max_avg_std(SU4_system, 'SU4 system')


        avg_r1_soa = self.print_min_max_avg_std(R1_SOA, 'R1 SOA')
        avg_r2_soa = self.print_min_max_avg_std(R2_SOA, 'R2 SOA')
        avg_r4_ub = self.print_min_max_avg_std(SU4_SOA, 'SU4 SOA')

        avg_accepts = sum(accepts)*1.0/len(accepts)
        avg_rejects = sum(rejects)*1.0/len(rejects)

        return avg_accepts, avg_rejects, avg_r2_ub, avg_r2_sys, avg_r2_soa

    def aggregate_scores(self, break_iteration):
        try:
            ub_scores = self.rows[0]
        except:
            return
        self.users = len(ub_scores[1:])/self.info_num

        #TODO: Change the initialization
        R1scores = [[] for _ in range(self.users)]
        R2scores = [[] for _ in range(self.users)]
        accepts = [[] for _ in range(self.users)]
        rejects = [[] for _ in range(self.users)]
        summaries = [[] for _ in range(self.users)]
        SU4scores = [[] for _ in range(self.users)]

        for iteration in range(0,len(self.rows)):
            if break_iteration !='last' and iteration == int(break_iteration)+1:
                break
            row = self.rows[iteration]
            for user in range(self.users):
                index = user*self.info_num
                val = row[1+index]
                if val != "":
                    R1scores[user].append(float(row[1+index]))
                    R2scores[user].append(float(row[2+index]))
                    SU4scores[user].append(float(row[3+index]))
                    accepts[user].append(int(row[4+index]))
                    rejects[user].append(int(row[5+index]))
                    summaries[user].append(str(row[6+index]))

        ub_reach, user_iterations = [], []
        for user in range(self.users):
            last = len(R1scores[user])
            user_iterations.append(last-1)
            if R1scores[user][0] <= R1scores[user][last-1] and R2scores[user][0] <= R2scores[user][last-1]:
                #print('Ub_score:', R1scores[user][0], R2scores[user][0])
                #print('Break point:', R1scores[user][last-1], R2scores[user][last-1])
                ub_reach.append(1)
                continue
            if R2scores[user][0] <= R2scores[user][last-1]:
                ub_reach.append(1)
                continue
            else:
                ub_reach.append(0)

        self.iterations.append(user_iterations)
        self.count_ub_reach.append(ub_reach)
        self.no_accepts.append(accepts)
        self.no_rejects.append(rejects)
        self.R1_scores.append(R1scores)
        self.R2_scores.append(R2scores)
        self.SU4_scores.append(SU4scores)
        self.summaries.append(summaries)

        """
        plt.subplot(211)
        for user in range(2):
            plt.plot(range(len(R2scores[user][1:])), len(R2scores[user][1:]) *[R2scores[user][0]], 'k--', label='UB%s' % (str(user)))
            plt.plot(range(len(R2scores[user][1:])), R2scores[user][1:], 'r', label='User %s' % (str(user)))

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
        """

def get_args():
    ''' This function parses and return arguments passed in'''

    parser = argparse.ArgumentParser(description='Results Aggregator')
    parser.add_argument('-d', '--data_set', type= str, help='Datset ex. DUC2004', required=True)
    parser.add_argument('-l', '--len_summary', type= str, help='Summary Size', required=False)
    parser.add_argument('-a', '--annotation', type= str, help='Annotation Type', required=False)

    args = parser.parse_args()
    data_set = args.data_set
    len_summary = args.len_summary
    annotation = args.annotation
    return data_set, len_summary, annotation

def plot_aggregate(labels, accepts_rejects, rejects, ub_score, sys_score, soa, break_iteration, filename):
    colors = ['g','b','r', '#8E4585']
    linestyles = ['->', '-o', '-', '-x']
    
    f, axis = plt.subplots(2, sharex=True, sharey=False, figsize=(4, 6))
    axis[0] = plt.subplot2grid((8, 12), (0, 0), rowspan=5, colspan=12)
    axis[1] = plt.subplot2grid((8, 12), (5, 0), rowspan=3, colspan=12)

    axis[0].plot(range(len(accepts_rejects[0][0:break_iteration])), len(accepts_rejects[0][0:break_iteration]) * [ub_score[0][0]], 'k--', label = 'Upper bound', linewidth=2)

    common_score = []
    for i in range(len(labels)):
        common_score.append(sys_score[i][0])

    initial_score = max(set(common_score), key=common_score.count)

    for i in range(len(labels)):
        sys_score[i][0] = initial_score

    for i in range(len(labels)):
        y = sys_score[i][0:break_iteration]
        axis[0].plot(range(len(y)), y, linestyles[i], color=colors[i], label='%s' % labels[i], linewidth=1.5)

    axis[0].set_ylabel('ROUGE 2', fontsize=15)
    axis[0].legend(loc="best", fontsize=12)
    axis[0].set_xticks(np.arange(0, break_iteration, 1))
    axis[0].set_autoscale_on(True)
    axis[0].grid(True)
    f.subplots_adjust(hspace=0.1)

    for i in range(len(labels)):
        y = accepts_rejects[i][1:break_iteration+1]
        axis[1].plot(range(len(y)), y, linestyles[i], color=colors[i], label='%s' % labels[i], linewidth=1.5)

    axis[1].grid(True)
    axis[1].set_xlabel("\# Iterations", fontsize=13)
    axis[1].set_ylabel('\# Feedbacks', fontsize=13)
    axis[1].set_xticks(np.arange(0, break_iteration, 1))
    axis[1].set_xticklabels(np.arange(0, break_iteration, 1))
    plt.tight_layout()
    savefig(filename)

if __name__ == '__main__':
    data_set, len_summary, annotation = get_args()

    methods = ['active_learning2', 'active_learning','ilp_feedback', 'accept_reject']
    labels = ['Active+','Active', 'Joint', 'Accept']
    total_iterations = 11

    
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
        "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
            r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
            ]
        }
    mpl.rcParams.update(pgf_with_latex)

    score_type = 'R2_score'

    accepts_rejects, rejects = [[] for i in range(len(methods))], [[] for i in range(len(methods))]
    ub_score, sys_score, soa_score = [[] for i in range(len(methods))], [[] for i in range(len(methods))], [[] for i in range(len(methods))]

    inter_topics = Set()
    for index, method in enumerate(methods):
        if len_summary!=None and annotation == None:
            data_path = '../data/scores/%s/%s_%s_%s' % (data_set, method, data_set, len_summary)
        if annotation!=None and len_summary!=None:
            data_path = '../data/scores/%s/%s_%s_%s_%s' % (data_set, method, annotation, data_set,len_summary)
        if annotation!=None and len_summary==None:
            data_path = '../data/scores/%s/%s_%s_%s' % (data_set, method, annotation, data_set)
        if annotation==None and len_summary==None:
            data_path = '../data/scores/%s/%s_%s' % (data_set, method, data_set)

        topics = [fileid[:-4] for fileid in os.listdir(data_path)]
        if index == 0:
            inter_topics = Set(topics)
        else:
            inter_topics = inter_topics.intersection(topics)

    file_ids = {}
    for break_iteration in range(1, total_iterations+2):
        for index, method in enumerate(methods):
            print('Method:%s, index:%d' % (method, index))
            if len_summary!=None and annotation == None:
                data_path = '../data/scores/%s/%s_%s_%s' % (data_set, method, data_set, len_summary)
            if annotation!=None and len_summary!=None:
                data_path = '../data/scores/%s/%s_%s_%s_%s' % (data_set, method, annotation, data_set,len_summary)
            if annotation!=None and len_summary==None:
                data_path = '../data/scores/%s/%s_%s_%s' % (data_set, method, annotation, data_set)
            if annotation==None and len_summary==None:
                data_path = '../data/scores/%s/%s_%s' % (data_set, method, data_set)

            aggregate = Aggregator()

            for fileid in os.listdir(data_path):
                filename = '%s/%s' % (data_path, fileid)
                topic = fileid[:-4]
                if topic not in inter_topics:
                    continue

                data_org_path = '../data/processed/%s/%s/summaries' % (data_set, topic)
                num_users = len(os.listdir(data_org_path))
                #print(filename)
                cluster_id = fileid[:-4]
                aggregate.load_data(filename, cluster_id, 6)
                aggregate.aggregate_scores(break_iteration)

            items = aggregate.print_results()
            avg_accepts, avg_rejects, avg_r2_ub, avg_r2_sys, avg_r2_soa = items
            accepts_rejects[index].append(avg_accepts)
            rejects[index].append(avg_rejects)
            ub_score[index].append(avg_r2_ub)
            sys_score[index].append(avg_r2_sys)
            soa_score[index].append(avg_r2_soa)

    filename = '%s' % (data_set)
    plot_aggregate(labels, accepts_rejects, rejects, ub_score, sys_score, soa_score, total_iterations, filename)

