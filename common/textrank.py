import numpy as np

class TextRank:
    def __init__(self, word_list, conv_thr=0.00001, damping_factor=0.85):
        self.word_ls = word_list
        self.conv_thr = conv_thr
        self.k = damping_factor
    
    def _similarity(self, count_matrix):
        similarity_matrix = \
            (count_matrix.T / count_matrix.sum(axis=1)).T
        
        return similarity_matrix
    
    def run(self):
        self.node = list(set(self.word_ls))
        matrix = \
            [[0 * i] * len(self.node) for i in range(len(self.node))]
        for i in range(len(self.word_ls)-1):
            matrix[self.node.index(self.word_ls[i])][self.node.index(self.word_ls[i+1])] = 1
            matrix[self.node.index(self.word_ls[i+1])][self.node.index(self.word_ls[i])] = 1
        count_matrix = np.array(matrix)
        
        # Calculate Similarity with Nodes
        similarity_matrix = self._similarity(count_matrix)
        
        # Calculate TextRank
        main_matrix = similarity_matrix
        new_matrix = np.zeros_like(matrix)
        diff = 1
        while diff > self.conv_thr:
            new_matrix = main_matrix
            node_sum = main_matrix.sum(axis=0) * self.k + (1 - self.k)
            main_matrix = (similarity_matrix.T * node_sum).T
            diff = np.sum(np.fabs(main_matrix - new_matrix))
        
        # Get TextRank Score
        self.textrank_score = main_matrix.sum(axis=1)


if __name__ == '__main__':
    sent = input("Input Text List : ").split()
    tr = TextRank(sent)
    tr.run()
    print("======= 핵심 키워드 =======")
    for i in range(len(tr.textrank_score)):
        print("{} : {:.4f}".format(tr.node[i], tr.textrank_score[i]))
