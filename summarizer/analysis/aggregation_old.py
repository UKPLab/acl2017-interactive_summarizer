import sys, os.path as path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from summarizer.utils.reader import read_csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import os
import argparse

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
        print mean, sigma

    def get_summaries(self, cluster_id, user_id):
        ub_summary = self.summaries[cluster_id][user_id][0]
        soa_summary = self.summaries[cluster_id][user_id][1]
        last = len(self.R2_scores[cluster_id][user_id])
        print self.R2_scores[cluster_id][user_id][0], self.R2_scores[cluster_id][user_id][1], self.R2_scores[cluster_id][user_id][last-1]  
        print 'UB summary:\n', ub_summary
        print 'SOA summary:\n', soa_summary
        
    def print_min_max_avg_std(self, array, tag):
        np_array = np.array(array)
        mean = np.mean(np_array)
        sigma = np.std(np_array)
        print '%s\nmin:%4f max:%4f avg:%4f std:%4f' % (tag, np.min(np_array), np.max(np_array), mean, sigma)
        '''
        if tag == 'R2 ub_soa diff:' or tag == 'R2 ub_last diff:':
            self.plot_distribution(mean, sigma, array)
        '''
    def print_results(self):
        ub_count, total_ub = 0.0, 0.0
        accepts, rejects, iterations = [], [], []
        R1_ub_last_diff, R2_ub_last_diff, SU4_ub_last_diff, other_accepts, other_rejects, other_iterations = [], [], [], [], [], []
        other_R1_ub_last_diff, other_R2_ub_last_diff, other_SU4_ub_last_diff = [], [], []
        reach_R1_ub_last_diff, reach_R2_ub_last_diff, reach_SU4_ub_last_diff = [], [], [] 
        R1_ub_soa_diff, R2_ub_soa_diff, SU4_ub_soa_diff = [], [], []
        R1_system, R2_system, SU4_system = [], [], []
        R1_UB, R2_UB, SU4_UB = [], [], []
        R1_SOA, R2_SOA, SU4_SOA = [], [], []
        R1_soa_last_diff, R2_soa_last_diff, SU4_soa_last_diff = [], [], []
        
        num_clusters = len(self.R1_scores)
        print num_clusters
        for cluster in range(num_clusters):
            no_users = len(self.count_ub_reach[cluster])
            for user in range(no_users):
                total_ub += 1
                last = len(self.R1_scores[cluster][user])
                
                index = np.argmax(np.array(self.R2_scores[cluster][user][1:last]))
                print self.R2_scores[cluster][user][1:last]
                index = index+1
                #index = last-1
                print index
                print self.R2_scores[cluster][user][index]
                
                accepts.append(sum(self.no_accepts[cluster][user][2:index]))
                rejects.append(sum(self.no_rejects[cluster][user][2:index]))
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
                    other_iterations.append(index)
                    other_R1_ub_last_diff.append(self.R1_scores[cluster][user][0]-self.R1_scores[cluster][user][index])
                    other_R2_ub_last_diff.append(self.R2_scores[cluster][user][0]-self.R2_scores[cluster][user][index])
            '''
            if self.count_ub_reach[cluster].count(1) == no_users:
                print 'All One Cluster: %s' % (self.clusters[cluster])
                print 'Iterations', self.iterations[cluster]
            if self.count_ub_reach[cluster].count(0) == no_users:
                print 'All Zero Cluster: %s' % (self.clusters[cluster])
                print 'Iterations', self.iterations[cluster]
            '''

        print 'No. of clusters:%d' % (num_clusters)
        print 'Total UB_Reach:%d/%d' % (ub_count, total_ub)
        print 'Total rejects avg:%d min,max:%d,%d' % (np.mean(np.array(rejects)), min(rejects),  max(rejects))
        print 'Total accepts avg:%d min,max:%d,%d' % (np.mean(np.array(accepts)), min(accepts), max(accepts))
        print 'Total iterations avg:%d min,max:%d,%d' % (np.mean(np.array(iterations)), min(iterations), max(iterations))
        
        max_diff_index = R2_soa_last_diff.index(min(R2_soa_last_diff))
        max_cluster, max_user = max_diff_index/self.users, max_diff_index%self.users
        #print 'Cluster with max difference:', self.clusters[max_cluster], max_user
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
        self.print_min_max_avg_std(R1_UB, 'R1 UB')
        self.print_min_max_avg_std(R2_UB, 'R2 UB')
        self.print_min_max_avg_std(SU4_UB, 'SU4 UB')
        
        self.print_min_max_avg_std(R1_system, 'R1 system')
        self.print_min_max_avg_std(R2_system, 'R2 system')
        self.print_min_max_avg_std(SU4_system, 'SU4 system')
        
        self.print_min_max_avg_std(R1_SOA, 'R1 SOA')
        self.print_min_max_avg_std(R2_SOA, 'R2 SOA')
        self.print_min_max_avg_std(SU4_SOA, 'SU4 SOA')
        
        
    def aggregate_scores(self, break_iteration):
        ub_scores = self.rows[0]
        self.users = len(ub_scores[1:])/self.info_num
        
        #TODO: Change the initialization
        R1scores = [[] for i in range(self.users)] 
        R2scores = [[] for i in range(self.users)] 
        SU4scores = [[] for i in range(self.users)]
        accepts = [[] for i in range(self.users)] 
        rejects = [[] for i in range(self.users)] 
        summaries = [[] for i in range(self.users)] 
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
                    #SU4scores[user].append(float(row[3+index]))
                    accepts[user].append(int(row[3+index]))
                    rejects[user].append(int(row[4+index]))
                    summaries[user].append(row[5+index])
                    SU4scores[user].append(float(0.0))
        
        ub_reach  = []
        for user in range(self.users):
            last = len(R1scores[user])
            if R1scores[user][0] <= R1scores[user][last-1] and R2scores[user][0] <= R2scores[user][last-1]:
                #print 'Ub_score:', R1scores[user][0], R2scores[user][0]
                #print 'Break point:', R1scores[user][last-1], R2scores[user][last-1]
                ub_reach.append(1)
                continue
            if R2scores[user][0] <= R2scores[user][last-1]:
                ub_reach.append(1)
                continue
            else:
                ub_reach.append(0)

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
    parser.add_argument('-w', '--writer_output', type= str, help='Writer Output Directory', required=True)
    parser.add_argument('-i', '--iteration', type= str, help='Writer Output Directory', required=False)

    args = parser.parse_args()
    woutput = args.writer_output
    iteration = args.iteration
    return woutput, iteration

if __name__ == '__main__':

    woutput, break_iteration = get_args()
    if break_iteration == None:
        break_iteration = 'last'
        
    aggregate = Aggregator()
    data_path = '%s' % (woutput)
    score_type = 'R2_score'
    for fileid in os.listdir(data_path):
        filename = '%s/%s' % (data_path, fileid)
        #print filename
        cluster_id = fileid[:-4]
        aggregate.load_data(filename, cluster_id, 5)
        #print cluster_id
        aggregate.aggregate_scores(break_iteration)
    aggregate.print_results()

