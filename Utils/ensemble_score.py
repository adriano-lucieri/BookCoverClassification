import numpy as np
import csv
import os

from sklearn.metrics import accuracy_score
from optparse import OptionParser


def read_csv(filepath):
    with open(filepath, 'r') as csvfile:
        content = csv.reader(csvfile, delimiter='|')
        
        temp_idx = []
        temp_val = []
        
        for row in content:
            temp_idx.append(int(row[1]))
            temp_val.append(int(row[1]))
            
    return np.array(temp_idx), np.array(temp_val)


# Command line options
parser = OptionParser()

parser.add_option("--ensemble", type="string", dest="ensemble", default='1',
				  help="Select ensemble setting to compute score.")
parser.add_option("--gt_labelpath", type="string", dest="gt_labelpath", default='./Data/title28cat-gt_labels.csv', 
				  help="Ground truth labels for test dataset.")

# Parse command line options
(options, args) = parser.parse_args()

if options.ensemble == '1':
	model_list = ['28cat-IncResV2-MSE', '28cat-IncResV2-Attention-Saliency', '28cat-IncResV2-Attention-TempSM', '28cat-IncResV2-Attention-Residual']
elif options.ensemble == '2':
	model_list = ['28cat-IncResV2-MSE', '28cat-IncResV2-Attention-Saliency', '28cat-IncResV2-Attention-TempSM', '28cat-IncResV2-Attention-Residual', '28cat-IncResV2-GAN07', '28cat-IncResV2-GAN10']
elif options.ensemble == '3':
	model_list = ['28cat-IncResV2-MSE', '28cat-IncResV2-Attention-Saliency', '28cat-IncResV2-Attention-TempSM', '28cat-IncResV2-Attention-Residual', '28cat-IncResV2-GAN07', '28cat-IncResV2-GAN10', '28cat-IncResV2-Augmentation', '28cat-IncResV2-random1', '28cat-IncResV2-random2']

_, gt_labels = read_csv(options.gt_labelpath)

predictions = np.zeros((len(gt_labels), len(model_list)))
for idx, model_name in enumerate(model_list):
	_, tmp_pred = read_csv(os.path.join('./Predictions/', model_name + '.csv'))
	predictions[:, idx] = tmp_pred

final_pred = []

for idx in range(len(gt_labels)):
    currentPredictions = predictions[idx]
    
    u, counts = np.unique(currentPredictions, return_counts=True)
    
    sort_idx   = np.argsort(counts)
    u = u[sort_idx]
    counts = counts[sort_idx]
    
    if len(u) == 1:
        final_pred.append(u[0])
    elif counts[-1] > counts[-2]:
        final_pred.append(u[-1])
    else:
        final_pred.append(u[-1])
    
final_pred = np.array(final_pred)
acc = accuracy_score(gt_labels, final_pred)
acc *= 100
print('Accuracy %.2f'%acc)

with open('./Scores.csv', 'a+') as f:
    writer = csv.writer(f, delimiter='|')
    writer.writerow(['Ensemble %s'%options.ensemble, acc])