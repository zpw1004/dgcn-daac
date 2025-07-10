import matplotlib.pyplot as plt
import numpy as np
import pickle
import itertools
import matplotlib.ticker as mtick
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Oranges):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.clip(cm, 0, 1)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1 if normalize else cm.max())
    ax.set_title(title, fontsize=16, pad=20)

    if normalize:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    else:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value = cm[i, j] * 100 if normalize else cm[i, j]
        text = format(value, fmt)
        ax.text(j, i, text, ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=7)

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    fig.tight_layout(pad=2)

    return fig, ax

confusion_matrix_path = ''

with open(confusion_matrix_path, 'rb') as f:
    conf_matrix = pickle.load(f)


attack_types = [
'SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D', 'PS', 'BS'
]



seen = set()
attack_types = [x for x in attack_types if not (x in seen or seen.add(x))]


fig, ax = plot_confusion_matrix(conf_matrix, classes=attack_types, normalize=True,
                                 title='DGCN-DAAC')
fig.savefig("DGCN-DAAC.jpg", dpi=600, bbox_inches='tight')
plt.show()
