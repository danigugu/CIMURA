import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class CIMURA_Plotter:
    def __init__():
        pass


    def plot_results(low_dim_csv, buffer=0.1, save_fig=None):

        df_low = pd.read_csv(low_dim_csv)
    
        def adjust_plot_limits(ax, data):
            x_min, x_max = data['dim1'].min(), data['dim1'].max()
            y_min, y_max = data['dim2'].min(), data['dim2'].max()

            x_range = x_max - x_min
            y_range = y_max - y_min
            max_range = max(x_range, y_range)

            x_mid = (x_max + x_min) / 2
            y_mid = (y_max + y_min) / 2
            x_lim = (x_mid - max_range / 2 - buffer, x_mid + max_range / 2 + buffer)
            y_lim = (y_mid - max_range / 2 - buffer, y_mid + max_range / 2 + buffer)

            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_aspect('equal', 'box')
            ax.grid(True, which='both', linestyle='-', linewidth='0.5')

        

        unique_groups = df_low['group'].unique()
        palette = sns.color_palette("viridis", len(unique_groups))
        group_palette = dict(zip(unique_groups, palette))

        fig, ax_scatter = plt.subplots(figsize=(10, 8))  # Adjust figure size as needed
        sns.scatterplot(data=df_low, x='dim1', y='dim2', hue='group', palette=group_palette, ax=ax_scatter)
        ax_scatter.set_xticklabels([])
        ax_scatter.set_yticklabels([])
        ax_scatter.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
        adjust_plot_limits(ax_scatter, df_low)

        plt.tight_layout()

        if save_fig is not None:
            plt.savefig(save_fig, dpi=300, bbox_inches='tight')
        else:
            plt.show()
