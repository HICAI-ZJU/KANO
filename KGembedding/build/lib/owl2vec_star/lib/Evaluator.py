from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class Evaluator:

    def __init__(self, valid_samples, test_samples, train_X, train_y):
        self.valid_samples = valid_samples
        self.test_samples = test_samples
        self.train_X = train_X
        self.train_y = train_y

    def evaluate(self, model, eva_samples):
        raise NotImplementedError('Function evaluate must be implemented!')

    # the simple one
    def run_random_forest(self):
        rf = RandomForestClassifier(n_estimators=200)
        rf.fit(self.train_X, self.train_y)
        rf_best = rf
        MRR, hits1, hits5, hits10 = self.evaluate(model=rf_best, eva_samples=self.test_samples)
        print('Testing, MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n\n' % (MRR, hits1, hits5, hits10))

    # the complete one
    """ 
     def run_random_forest(self):
         rf_best = None
         rf_best_mrr = 0.0
         rf_best_tree = 0
         for tree_n in [100, 200]:
             rf = RandomForestClassifier(n_estimators=tree_n)
             rf.fit(self.train_X, self.train_y)
             mrr, _, _, _ = self.evaluate(model=rf, eva_samples=self.valid_samples)
             print('Random forest, tree_n: %d, valid MRR: %.3f' % (tree_n, mrr))
             if mrr > rf_best_mrr:
                 rf_best_mrr = mrr
                 rf_best = rf
                 rf_best_tree = tree_n
         print('\nSelected random forest, tree_n: %d, validation MRR: %.3f' % (rf_best_tree, rf_best_mrr))
         MRR, hits1, hits5, hits10 = self.evaluate(model=rf_best, eva_samples=self.test_samples)
         print('Testing, MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n\n' %
               (MRR, hits1, hits5, hits10))
    """

    def run_mlp(self):
        mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=200)
        mlp.fit(self.train_X, self.train_y)
        mlp_best = mlp
        MRR, hits1, hits5, hits10 = self.evaluate(model=mlp_best, eva_samples=self.test_samples)
        print('Testing, MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n\n' % (MRR, hits1, hits5, hits10))

    # the complete one
    """
    def run_mlp(self):
        mlp_best = None
        mlp_best_mrr = 0.0
        mlp_best_hidden = 0
        for hidden_n in [50, 100, 150, 200, 250]:
            mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=hidden_n)
            mlp.fit(self.train_X, self.train_y)
            mrr, _, _, _ = self.evaluate(model=mlp, eva_samples=self.valid_samples)
            print('MLP, tree_n: %d, valid MRR: %.3f' % (hidden_n, mrr))
            if mrr > mlp_best_mrr:
                mlp_best_mrr = mrr
                mlp_best = mlp
                mlp_best_hidden = hidden_n
        print('\nSelected MLP, hidden_n: %d, validation MRR: %.3f' % (mlp_best_hidden, mlp_best_mrr))
        MRR, hits1, hits5, hits10 = self.evaluate(model=mlp_best, eva_samples=self.test_samples)
        print('Testing, MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n\n' %
              (MRR, hits1, hits5, hits10))
    """

    def run_logistic_regression(self):
        lr = LogisticRegression(random_state=0)
        lr.fit(self.train_X, self.train_y)
        lr_best = lr
        MRR, hits1, hits5, hits10 = self.evaluate(model=lr_best, eva_samples=self.test_samples)
        print('Testing, MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n\n' % (MRR, hits1, hits5, hits10))

    def run_svm(self):
        m = svm.SVC(probability=True)
        m.fit(self.train_X, self.train_y)
        m_best = m
        MRR, hits1, hits5, hits10 = self.evaluate(model=m_best, eva_samples=self.test_samples)
        print('Testing, MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n\n' % (MRR, hits1, hits5, hits10))

    def run_linear_svc(self):
        lin_clf = svm.LinearSVC()
        m = CalibratedClassifierCV(lin_clf)
        m.fit(self.train_X, self.train_y)
        m_best = m
        MRR, hits1, hits5, hits10 = self.evaluate(model=m_best, eva_samples=self.test_samples)
        print('Testing, MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n\n' % (MRR, hits1, hits5, hits10))

    def run_decision_tree(self):
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(self.train_X, self.train_y)
        m_best = dt
        MRR, hits1, hits5, hits10 = self.evaluate(model=m_best, eva_samples=self.test_samples)
        print('Testing, MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n\n' % (MRR, hits1, hits5, hits10))

    def run_sgd_log(self):
        clf = make_pipeline(StandardScaler(), SGDClassifier(loss='log'))
        clf.fit(self.train_X, self.train_y)
        m_best = clf
        MRR, hits1, hits5, hits10 = self.evaluate(model=m_best, eva_samples=self.test_samples)
        print('Testing, MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n\n' % (MRR, hits1, hits5, hits10))
