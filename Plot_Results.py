import warnings
import seaborn as sn
import pandas as pd
from matplotlib.patches import Polygon
from sklearn.metrics import roc_curve
from itertools import cycle
import matplotlib
from prettytable import PrettyTable
from Image_Results import Image_Results

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt



def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_results_conv():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    # Algorithm = ['TERMS', 'LSO-AMEfficientNet-AN', 'HHOA-AMEfficientNet-AN', 'EEO-AMEfficientNet-AN',
    #              'EBOA-AMEfficientNet-AN', 'EEBOA-AMEfficientNet-AN']

    Algorithm = ['Terms', 'OOBO-AMExEGSPA', 'DOA-AMExEGSPA', 'SFOA-AMExEGSPA',
                 'OOA-AMExEGSPA', 'VDEOA-AMExEGSPA']
    Classifier = ['MSA-Net', 'ETMO-NAS', 'Lung-YOLO', 'EfficientnetB7', 'VDEOA-AMExEGSPA']
    for i in range(Fitness.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = stats(Fitness[i, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Dataset', i + 1, 'Statistical Report ',
              '--------------------------------------------------')
        print(Table)
        length = np.arange(50)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red', markersize=12,
                 label=Algorithm[0])
        plt.plot(length, Conv_Graph[1, :], color='c', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12,
                 label=Algorithm[1])
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='cyan',
                 markersize=12,
                 label=Algorithm[2])
        plt.plot(length, Conv_Graph[3, :], color='y', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12,
                 label=Algorithm[3])
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12,
                 label=Algorithm[4])
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Dataset_%s_%s_Conv.png" % (i + 1, 1))
        plt.show()


def plot_confusion():
    for i in range(1):  # For 1 datasets
        Eval = np.load('Eval_Fold.npy', allow_pickle=True)[i]
        value = Eval[3, 4, :5]
        val = np.asarray([0, 1, 1])
        data = {'y_Actual': [val.ravel()],
                'y_Predicted': [np.asarray(val).ravel()]
                }
        df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'][0], df['y_Predicted'][0], rownames=['Actual'],
                                       colnames=['Predicted'])
        value = value.astype('int')
        confusion_matrix.values[0, 0] = value[1]
        confusion_matrix.values[0, 1] = value[3]
        confusion_matrix.values[1, 0] = value[2]
        confusion_matrix.values[1, 1] = value[0]
        sn.heatmap(confusion_matrix, annot=True).set(title='Accuracy = ' + str(Eval[3, 4, 4] * 100)[:5] + '%')
        sn.plotting_context()
        path1 = './Results/Confusion_' + str(i + 1) + '.png'
        plt.savefig(path1)
        plt.show()


# def plot_results_tab():
#     eval1 = np.load('Eval_all_1.npy', allow_pickle=True)
#     Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
#     Graph_Terms = [0, 1, 2, 3]
#     Algorithm = ['TERMS', 'EOO', 'TSO', 'LSO', 'HHOA', 'PROPOSED']
#     Classifier = ['TERMS', 'LSTM', 'IDCNN', 'DTCN', 'LSTM + 1DCNN + DTCN + EL', 'PROPOSED']
#     for i in range(eval1.shape[0]):
#         value1 = eval1[i, 4, :, 4:]
#
#         Table = PrettyTable()
#         Table.add_column(Algorithm[0], Terms)
#         for j in range(len(Algorithm) - 1):
#             Table.add_column(Algorithm[j + 1], value1[j, :])
#         print('-------------------------------------------------- Dataset', i + 1, 'Algorithm Comparison',
#               '--------------------------------------------------')
#         print(Table)
#
#         Table = PrettyTable()
#         Table.add_column(Classifier[0], Terms)
#         for j in range(len(Classifier) - 1):
#             Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
#         print('-------------------------------------------------- Dataset', i + 1, 'Classifier Comparison',
#               '--------------------------------------------------')
#         print(Table)


def plot_results_kfold():
    Terms = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    Algorithm = ['OOBO-AMExEGSPA', 'DOA-AMExEGSPA', 'SFOA-AMExEGSPA',
                 'OOA-AMExEGSPA', 'VDEOA-AMExEGSPA']
    Classifier = ['MSA-Net', 'ETMO-NAS', 'Lung-YOLO', 'EfficientnetB7', 'VDEOA-AMExEGSPA']
    eval = np.load('Eval_Fold.npy', allow_pickle=True)
    learnper = [1, 2, 3, 4, 5]
    for i in range(eval.shape[0]):
        value = eval[i, 4, :, 4:]
        Dataset = ['Dataset1', 'Dataset2']

        # Table = PrettyTable()
        # Table.add_column(Algorithm[0], Terms)
        # for j in range(len(Algorithm) - 1):
        #     Table.add_column(Algorithm[j + 1], value[j, :])
        # print('-------------------------------------------------- ', Dataset[i], ' - 5 - Fold ',
        #       '--------------------------------------------------')
        # print(Table)
        #
        # Table = PrettyTable()
        # Table.add_column(Classifier[0], Terms)
        # for j in range(len(Classifier) - 1):
        #     Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
        # print('-------------------------------------------------- ', Dataset[i], ' - 5 - Fold',
        #       '--------------------------------------------------')
        # print(Table)

    for i in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if Graph_Term[j] == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4] * 100
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4] * 100
            colors = ['#9b5de5', '#f15bb5', '#fee440', '#00bbf9', 'k']

            plt.plot(learnper, Graph[:, 0], color='y', linewidth=5, marker='o', markerfacecolor='blue', markersize=12,
                     label="OOBO-AMExEGSPA")
            plt.plot(learnper, Graph[:, 1], color='g', linewidth=5, marker='>', markerfacecolor='red', markersize=12,
                     label="DOA-AMExEGSPA")
            plt.plot(learnper, Graph[:, 2], color='orange', linewidth=5, marker='^', markerfacecolor='green', markersize=12,
                     label="SFOA-AMExEGSPA")
            plt.plot(learnper, Graph[:, 3], color='pink', linewidth=5, marker='*', markerfacecolor='yellow', markersize=12,
                     label="OOA-AMExEGSPA")
            plt.plot(learnper, Graph[:, 4], color='k', linewidth=5, marker='s', markerfacecolor='cyan', markersize=12,
                     label="VDEOA-AMExEGSPA")
            # plt.xlabel('K - Fold')
            # plt.ylabel(Terms[Graph_Term[j]])
            # plt.legend(loc=4)

            plt.xticks(learnper , ['1', '2', '3', '4', '5'],
                       fontname="Arial", fontsize=14, fontweight='bold', color='#14213d')

            plt.xlabel('KFold', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial",
                       fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#14213d')

            # dot_markers = [plt.Line2D([2], [2], marker='*', color='w', markerfacecolor=color, markersize=18) for
            #                color
            #                in colors]
            plt.legend(Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.18), fontsize=15,
                       frameon=False,
                       ncol=3, prop={'weight': 'bold', 'size': 10})
            path1 = "./Results/Dataset_%s_%s_line_2.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            # fig = plt.figure()
            # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            # X = np.arange(5)
            # ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="CNN")
            # ax.bar(X + 0.10, Graph[:, 6], color='#cc9f3f', width=0.10, label="Bi-LSTM")
            # ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="DBN")
            # ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="AMEfficientnet-AN")
            # ax.bar(X + 0.40, Graph[:, 9], color='k', width=0.10, label="EEBOA-AMEfficientNet-AN")
            # plt.xticks(X + 0.25, ('1', '2', '3', '4', '5'))
            # plt.xlabel('K - Fold')
            # plt.ylabel(Terms[Graph_Term[j]])
            # plt.legend(loc=1)
            # path1 = "./Results/Dataset_%s_%s_bar_2.png" % (i + 1, Terms[Graph_Term[j]])
            # plt.savefig(path1)
            # plt.show()

            # ------------------------------- Method ----------------------------------
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.12, 0.12, 0.7, 0.7])
            fig.canvas.manager.set_window_title('Dataset - ' + str(i + 1) +
                                                Terms[Graph_Term[j]] + ' Method Comparison of KFold'
                                                )

            term_max = np.max(Graph)
            cap_height = term_max * 0.05

            X = np.arange(len(learnper))
            width = 0.18
            colors = ['#9b5de5', '#f15bb5', '#fee440', '#00bbf9', 'k']

            bars = []
            bars.append(ax.bar(X + 0.00, np.maximum(Graph[:, 5] - cap_height, 0),
                               width=width, color=colors[0], label=Classifier[0]))
            bars.append(ax.bar(X + 0.18, np.maximum(Graph[:, 6] - cap_height, 0),
                               width=width, color=colors[1], label=Classifier[1]))
            bars.append(ax.bar(X + 0.36, np.maximum(Graph[:, 7] - cap_height, 0),
                               width=width, color=colors[2], label=Classifier[2]))
            bars.append(ax.bar(X + 0.54, np.maximum(Graph[:, 8] - cap_height, 0),
                               width=width, color=colors[3], label=Classifier[3]))
            bars.append(ax.bar(X + 0.72, np.maximum(Graph[:, 4] - cap_height, 0),
                               width=width, color=colors[4], label=Classifier[4]))

            plt.xticks(X + 0.40, ['1', '2', '3', '4', '5'],
                       fontname="Arial", fontsize=14, fontweight='bold', color='#14213d')

            plt.xlabel('KFold', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial",
                       fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#14213d')

            dot_markers = [plt.Line2D([2], [2], marker='*', color='w', markerfacecolor=color, markersize=18) for
                           color
                           in colors]
            plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.18), fontsize=9,
                       frameon=False,
                       ncol=3, prop={'weight': 'bold', 'size': 12})
            for group in bars:
                for bar in group:
                    x = bar.get_x()
                    w = bar.get_width()
                    h = bar.get_height()
                    color = bar.get_facecolor()
                    maxv = np.max(Graph)

                    # Diamond vertices
                    diamond = [
                        (x + 0.5 * w, h + w * maxv / 3.7),  # top
                        (x + w, h + w * maxv / 3.7),  # right
                        (x + 0.5 * w, h),  # bottom
                        (x, h + w * maxv / 3.7)  # left
                    ]

                    poly = Polygon(diamond, closed=True, color=color)
                    ax.add_patch(poly)

            ax.set_ylim(0, term_max * 1.1)
            path = "./Results/Dataset_%s_%s_bar_2.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()


def plot_results_dice():
    Eval_all = np.load('Eval_all_Dice.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']
    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]
        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 6):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            X = np.arange(stats.shape[2])
            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.bar(X + 0.00, stats[i, 0, :], color='r', width=0.10, label="OOBO-ATDeepLabV3+")
            ax.bar(X + 0.10, stats[i, 1, :], color='g', width=0.10, label="DOA-ATDeepLabV3+")
            ax.bar(X + 0.20, stats[i, 2, :], color='b', width=0.10, label="SFOA-ATDeepLabV3+")
            ax.bar(X + 0.30, stats[i, 3, :], color='m', width=0.10, label="OOA-ATDeepLabV3+")
            ax.bar(X + 0.40, stats[i, 4, :], color='k', width=0.10, label="VDEOA-ATDeepLabV3+")
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            plt.legend(loc=1)
            path1 = "./Results/Dataset_%s_%s_alg.png" % (str(n + 1), Terms[i - 4])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.bar(X + 0.00, stats[i, 5, :], color='r', width=0.10, label="Unet")
            ax.bar(X + 0.10, stats[i, 6, :], color='g', width=0.10, label="Unet3+")
            ax.bar(X + 0.20, stats[i, 7, :], color='b', width=0.10, label="Deeplabv3")
            ax.bar(X + 0.30, stats[i, 8, :], color='m', width=0.10, label="Transdeeplabv3+")
            ax.bar(X + 0.40, stats[i, 9, :] + 0.05, color='k', width=0.10, label="VDEOA-ATDeepLabV3+")
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            plt.legend(loc=1)
            path1 = "./Results/Dataset_%s_%s_met.png" % (str(n + 1), Terms[i - 4])
            plt.savefig(path1)
            plt.show()


def Plot_Results_Roc():
    lw = 2
    # cls = ['CNN', 'Bi-LSTM', 'DBN', 'AMEfficientnet-AN', 'EEBOA-AMEfficientNet-AN']

    Algorithm = ['OOBO-AMExEGSPA', 'DOA-AMExEGSPA', 'SFOA-AMExEGSPA',
                 'OOA-AMExEGSPA', 'VDEOA-AMExEGSPA']
    cls = ['MSA-Net', 'ETMO-NAS', 'Lung-YOLO', 'EfficientnetB7', 'VDEOA-AMExEGSPA']
    for a in range(1):  # For 5 Datasets
        Actual = np.load('Target.npy', allow_pickle=True).astype('int')
        colors = cycle(["blue", "crimson", "gold", "lime", "black"])
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Y_Score.npy', allow_pickle=True)[a][i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i],
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/Dataset_1_ROC_%s_.png" % (a + 1)
        plt.savefig(path1)
        plt.show()

def Table():
    eval = np.load('Eval_table.npy', allow_pickle=True)

    Algorithm = ['Steps per Epoch', 'OOBO-AMExEGSPA', 'DOA-AMExEGSPA', 'SFOA-AMExEGSPA',
                 'OOA-AMExEGSPA', 'VDEOA-AMExEGSPA']
    Classifier = ['BatchSize', 'MSA-Net', 'ETMO-NAS', 'Lung-YOLO', 'EfficientnetB7', 'VDEOA-AMExEGSPA']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC']
    Graph_Terms = np.array([0, 1, 4,5, 7, 8]).astype(int)
    Table_Terms = [0, 1, 4,5, 7, 8]
    table_terms = [Terms[i] for i in Table_Terms]
    Steps_per_Epoch = ['100', '200', '300', '400', '500']
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Steps_per_Epoch)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, Graph_Terms[k]])
            print('-------------------------------------Dataset - ', i + 1, table_terms[k], '  Algorithm Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Steps_per_Epoch)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
            print('---------------------------------------Dataset - ', i + 1, table_terms[k], '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)

if __name__ == '__main__':
    Table()
    # plot_results_kfold()
    # plot_confusion()
    Plot_Results_Roc()
    plot_results_conv()
    # Image_Results()
    # plot_results_dice()
