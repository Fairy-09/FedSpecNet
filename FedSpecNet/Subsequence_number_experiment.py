import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def plot_subsequence_result(data_path):
        data = pd.read_csv(data_path)
        x = data['Number-of-Subsequence'][:30]
        y_house1_mae = data['MAE'][:30]
        y_house2_mae = data['MAE-2'][:30]
        y_house3_mae = data['MAE-3'][:30]
        y_house4_mae = data['MAE-4'][:30]
        y_house5_mae = data['MAE-5'][:30]

        y_house1_mae1 = data['MAPE'][:30]
        y_house1_mae2 = data['MAPE-2'][:30]
        y_house1_mae3 = data['MAPE-3'][:30]
        y_house1_mae4 = data['MAPE-4'][:30]
        y_house1_mae5 = data['MAPE-5'][:30]

        y_house2_mae1 = data['RMSE'][:30]
        y_house2_mae2 = data['RMSE-2'][:30]
        y_house2_mae3 = data['RMSE-3'][:30]
        y_house2_mae4 = data['RMSE-4'][:30]
        y_house2_mae5 = data['RMSE-5'][:30]

        y_house3_mae1 = data['R2'][:30]
        y_house3_mae1 = [max(0, value) for value in y_house3_mae1]
        y_house3_mae2 = data['R2-2'][:30]
        y_house3_mae2 = [max(0, value) for value in y_house3_mae2]
        y_house3_mae3 = data['R2-3'][:30]
        y_house3_mae3 = [max(0, value) for value in y_house3_mae3]
        y_house3_mae4 = data['R2-4'][:30]
        y_house3_mae4 = [max(0, value) for value in y_house3_mae4]
        y_house3_mae5 = data['R2-5'][:30]
        y_house3_mae5 = [max(0, value) for value in y_house3_mae5]

        plt.figure(figsize=(20, 12))

        ahead_itf_idx = [14, 8, 28, 20, 18]  #
        end_itf_idx = [16, 10, 30, 22, 20]

        plt.plot(x, y_house1_mae, label='house1', color='orange', linestyle='-', marker='o', linewidth=2.5)
        plt.plot(x, y_house2_mae, label='house2', color='red', linestyle='-', marker='s', linewidth=2.5)
        plt.plot(x, y_house3_mae, label='house3', color='brown', linestyle='-', marker='h', linewidth=2.5)
        plt.plot(x, y_house4_mae, label='house4', color='green', linestyle='-', marker='d', linewidth=2.5)
        plt.plot(x, y_house5_mae, label='house5', color='blue', linestyle='-', marker='d', linewidth=2.5)
        y_range = plt.ylim()  
        y = y_range[1] * 0.95

        plt.axvline(x=ahead_itf_idx[0], color='k', linestyle='--')
        plt.axvline(x=end_itf_idx[0], color='k', linestyle='--')

        plt.axvline(x=ahead_itf_idx[1], color='k', linestyle='--')
        plt.axvline(x=end_itf_idx[1], color='k', linestyle='--')

        plt.axvline(x=ahead_itf_idx[2], color='k', linestyle='--')
        plt.axvline(x=end_itf_idx[2], color='k', linestyle='--')

        plt.axvline(x=ahead_itf_idx[3], color='k', linestyle='--')
        plt.axvline(x=end_itf_idx[3], color='k', linestyle='--')

        plt.axvline(x=ahead_itf_idx[4], color='k', linestyle='--')
        plt.axvline(x=end_itf_idx[4], color='k', linestyle='--')

        plt.axvspan(ahead_itf_idx[0], end_itf_idx[0], color='orange', alpha=0.5)  
        plt.text((ahead_itf_idx[0] + end_itf_idx[0]) / 2, y, 'house1_range',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')

        plt.axvspan(ahead_itf_idx[1], end_itf_idx[1], color='red', alpha=0.5)  
        plt.text((ahead_itf_idx[1] + end_itf_idx[1]) / 2, y, 'house2_range',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')

        plt.axvspan(ahead_itf_idx[2], end_itf_idx[2], color='brown', alpha=0.5)  
        plt.text((ahead_itf_idx[2] + end_itf_idx[2]) / 2, y, 'house3_range',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')

        plt.axvspan(ahead_itf_idx[3], end_itf_idx[3], color='green', alpha=0.5)  
        plt.text((ahead_itf_idx[3] + end_itf_idx[3]) / 2, y, 'house4_range',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')

        plt.axvspan(ahead_itf_idx[4], end_itf_idx[4], color='blue', alpha=0.5)  
        plt.text((ahead_itf_idx[4] + end_itf_idx[4]) / 2, y, 'house5_range',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')
        plt.xlabel('Number-of-Subsequence', fontsize=18)
        plt.legend(loc='upper right',prop={'size': 10})
        plt.grid(False)
        plt.ylabel('MAE(W)', fontsize=18)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(20, 12))
        plt.plot(x, y_house1_mae1, label='house1', color='orange', linestyle='-', marker='o', linewidth=2.5)
        plt.plot(x, y_house1_mae2, label='house2', color='red', linestyle='-', marker='s', linewidth=2.5)
        plt.plot(x, y_house1_mae3, label='house3', color='brown', linestyle='-', marker='h', linewidth=2.5)
        plt.plot(x, y_house1_mae4, label='house4', color='green', linestyle='-', marker='d', linewidth=2.5)
        plt.plot(x, y_house1_mae5, label='house5', color='blue', linestyle='-', marker='d', linewidth=2.5)
        y_range = plt.ylim()  
        y = y_range[1] * 0.95

        plt.axvline(x=ahead_itf_idx[0], color='k', linestyle='--')
        plt.axvline(x=end_itf_idx[0], color='k', linestyle='--')

        plt.axvline(x=ahead_itf_idx[1], color='k', linestyle='--')
        plt.axvline(x=end_itf_idx[1], color='k', linestyle='--')

        plt.axvline(x=ahead_itf_idx[2], color='k', linestyle='--')
        plt.axvline(x=end_itf_idx[2], color='k', linestyle='--')

        plt.axvline(x=ahead_itf_idx[3], color='k', linestyle='--')
        plt.axvline(x=end_itf_idx[3], color='k', linestyle='--')

        plt.axvline(x=ahead_itf_idx[4], color='k', linestyle='--')
        plt.axvline(x=end_itf_idx[4], color='k', linestyle='--')

        plt.axvspan(ahead_itf_idx[0], end_itf_idx[0], color='orange', alpha=0.5)  
        plt.text((ahead_itf_idx[0] + end_itf_idx[0]) / 2, y, 'house1_range',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')

        plt.axvspan(ahead_itf_idx[1], end_itf_idx[1], color='red', alpha=0.5)  
        plt.text((ahead_itf_idx[1] + end_itf_idx[1]) / 2, y, 'house2_range',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')

        plt.axvspan(ahead_itf_idx[2], end_itf_idx[2], color='brown', alpha=0.5)  
        plt.text((ahead_itf_idx[2] + end_itf_idx[2]) / 2, y, 'house3_range',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')

        plt.axvspan(ahead_itf_idx[3], end_itf_idx[3], color='green', alpha=0.5)  
        plt.text((ahead_itf_idx[3] + end_itf_idx[3]) / 2, y, 'house4_range',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')

        plt.axvspan(ahead_itf_idx[4], end_itf_idx[4], color='blue', alpha=0.5)  
        plt.text((ahead_itf_idx[4] + end_itf_idx[4]) / 2, y, 'house5_range',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')
        plt.xlabel('Number-of-Subsequence', fontsize=18)
        plt.legend(loc='upper right',prop={'size': 10})
        plt.grid(False)
        plt.ylabel('MAPE(%)', fontsize=18)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(20, 12))
        plt.plot(x, y_house2_mae1, label='house1', color='orange', linestyle='-', marker='o', linewidth=2.5)
        plt.plot(x, y_house2_mae2, label='house2', color='red', linestyle='-', marker='s', linewidth=2.5)
        plt.plot(x, y_house2_mae3, label='house3', color='brown', linestyle='-', marker='h', linewidth=2.5)
        plt.plot(x, y_house2_mae4, label='house4', color='green', linestyle='-', marker='d', linewidth=2.5)
        plt.plot(x, y_house2_mae5, label='house5', color='blue', linestyle='-', marker='d', linewidth=2.5)
        y_range = plt.ylim()  
        y = y_range[1] * 0.95

        plt.axvline(x=ahead_itf_idx[0], color='k', linestyle='--')
        plt.axvline(x=end_itf_idx[0], color='k', linestyle='--')

        plt.axvline(x=ahead_itf_idx[1], color='k', linestyle='--')
        plt.axvline(x=end_itf_idx[1], color='k', linestyle='--')

        plt.axvline(x=ahead_itf_idx[2], color='k', linestyle='--')
        plt.axvline(x=end_itf_idx[2], color='k', linestyle='--')

        plt.axvline(x=ahead_itf_idx[3], color='k', linestyle='--')
        plt.axvline(x=end_itf_idx[3], color='k', linestyle='--')

        plt.axvline(x=ahead_itf_idx[4], color='k', linestyle='--')
        plt.axvline(x=end_itf_idx[4], color='k', linestyle='--')

        plt.axvspan(ahead_itf_idx[0], end_itf_idx[0], color='orange', alpha=0.5)  
        plt.text((ahead_itf_idx[0] + end_itf_idx[0]) / 2, y, 'house1_range',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')

        plt.axvspan(ahead_itf_idx[1], end_itf_idx[1], color='red', alpha=0.5)  
        plt.text((ahead_itf_idx[1] + end_itf_idx[1]) / 2, y, 'house2_range',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')

        plt.axvspan(ahead_itf_idx[2], end_itf_idx[2], color='brown', alpha=0.5)  
        plt.text((ahead_itf_idx[2] + end_itf_idx[2]) / 2, y, 'house3_range',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')

        plt.axvspan(ahead_itf_idx[3], end_itf_idx[3], color='green', alpha=0.5)  
        plt.text((ahead_itf_idx[3] + end_itf_idx[3]) / 2, y, 'house4_range',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')

        plt.axvspan(ahead_itf_idx[4], end_itf_idx[4], color='blue', alpha=0.5)  
        plt.text((ahead_itf_idx[4] + end_itf_idx[4]) / 2, y, 'house5_range',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')
        plt.xlabel('Number-of-Subsequence', fontsize=18)
        plt.legend(loc='upper right',prop={'size': 10})
        plt.grid(False)
        plt.ylabel('RMSE(W)', fontsize=18)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(20, 12))
        plt.plot(x, y_house3_mae1, label='house1', color='orange', linestyle='-', marker='o', linewidth=2.5)
        plt.plot(x, y_house3_mae2, label='house2', color='red', linestyle='-', marker='s', linewidth=2.5)
        plt.plot(x, y_house3_mae3, label='house3', color='brown', linestyle='-', marker='h', linewidth=2.5)
        plt.plot(x, y_house3_mae4, label='house4', color='green', linestyle='-', marker='d', linewidth=2.5)
        plt.plot(x, y_house3_mae5, label='house5', color='blue', linestyle='-', marker='d', linewidth=2.5)
        y_range = plt.ylim()  
        y = y_range[1] * 0.95

        plt.axvline(x=ahead_itf_idx[0], color='k', linestyle='--')
        plt.axvline(x=end_itf_idx[0], color='k', linestyle='--')

        plt.axvline(x=ahead_itf_idx[1], color='k', linestyle='--')
        plt.axvline(x=end_itf_idx[1], color='k', linestyle='--')

        plt.axvline(x=ahead_itf_idx[2], color='k', linestyle='--')
        plt.axvline(x=end_itf_idx[2], color='k', linestyle='--')

        plt.axvline(x=ahead_itf_idx[3], color='k', linestyle='--')
        plt.axvline(x=end_itf_idx[3], color='k', linestyle='--')

        plt.axvline(x=ahead_itf_idx[4], color='k', linestyle='--')
        plt.axvline(x=end_itf_idx[4], color='k', linestyle='--')

        plt.axvspan(ahead_itf_idx[0], end_itf_idx[0], color='orange', alpha=0.5)  
        plt.text((ahead_itf_idx[0] + end_itf_idx[0]) / 2, y, 'house1_range',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')

        plt.axvspan(ahead_itf_idx[1], end_itf_idx[1], color='red', alpha=0.5)  
        plt.text((ahead_itf_idx[1] + end_itf_idx[1]) / 2, y, 'house2_range',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')

        plt.axvspan(ahead_itf_idx[2], end_itf_idx[2], color='brown', alpha=0.5)  
        plt.text((ahead_itf_idx[2] + end_itf_idx[2]) / 2, y, 'house3_range',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')

        plt.axvspan(ahead_itf_idx[3], end_itf_idx[3], color='green', alpha=0.5)  
        plt.text((ahead_itf_idx[3] + end_itf_idx[3]) / 2, y, 'house4_range',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')

        plt.axvspan(ahead_itf_idx[4], end_itf_idx[4], color='blue', alpha=0.5)  
        plt.text((ahead_itf_idx[4] + end_itf_idx[4]) / 2, y, 'house5_range',
                 horizontalalignment='center', verticalalignment='top', fontsize=10, color='black')
        plt.xlabel('Number-of-Subsequence', fontsize=18)
        plt.legend(loc='upper right',prop={'size': 10})
        plt.grid(False)
        plt.ylabel('R2', fontsize=18)
        plt.tight_layout()
        plt.show()
