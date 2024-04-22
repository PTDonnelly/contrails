import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path

class Dataset:
    def __init__(self, base_path, start_date, end_date):
        self.base_path = Path(base_path)
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.data = None
        self.load_and_process_data()

    def load_and_process_data(self):
        daily_averages = []
        current_date = self.start_date
        while current_date <= self.end_date:
            year = current_date.strftime('%Y')
            month = current_date.strftime('%m')
            day = current_date.strftime('%d')

            file_path = self.base_path / f"{year}/{month}/{day}/spectra_and_cloud_products_binned.csv"
            if file_path.exists():
                df = pd.read_csv(file_path, sep='\t')
                df['Date'] = pd.to_datetime(df['Date'])
                daily_average = df[['OLR_mean']].mean()
                daily_average['Date'] = current_date
                daily_averages.append(daily_average)
            current_date += timedelta(days=1)
        
        self.data = pd.DataFrame(daily_averages)
        self.data.set_index('Date', inplace=True)
        return

class DataPlotter:
    def __init__(self, dataset):
        self.dataset = dataset

    def add_grey_box(self, ax, df, plot_type):
        """
        Adds grey boxes to the plot for every other year.

        Parameters:
        - ax (matplotlib.axes.Axes): The axes object to add the grey boxes to.
        - df (pd.DataFrame): DataFrame with 'Year' column.
        """
        unique_years = sorted(df['Year'].unique())
        for i, year in enumerate(unique_years):
            
            if plot_type == 'violin':
                if i % 2 == 0:
                    ax.axvspan(i-0.5, i+0.5, color='grey', alpha=0.2, zorder=0)
            elif plot_type == 'line':
                if i % 2 == 0:
                    ax.axvspan(year, year+1, color='grey', alpha=0.2, zorder=0)
        return ax
    
    @staticmethod
    def make_date_axis_continuous(df, number_of_months=3, number_of_days=0):
        """For converting a Date index in DataFrame to a continuous numerical axis for plotting."""
        df['Year-Month-Day'] = df.index.year + ((df.index.month - number_of_months) / number_of_months) + ((df.index.day - number_of_days)/ 100)
        return df

    def plot_yearly_trend(self):
        plt.figure(figsize=(12, 6))
        sns.violinplot(x=self.dataset.data.index.year, y='OLR_mean', data=self.dataset.data)
        plt.title('Yearly Violin Plot of Daily Average OLR')
        plt.xlabel('Year')
        plt.ylabel('Daily Average OLR')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset.base_path, f"olr_yearly_trend.png"), bbox_inches='tight')
        plt.close()

    def plot_monthly_trend(self):
        plt.figure(figsize=(12, 6))
        sns.violinplot(x=self.dataset.data.index.month, y='OLR_mean', data=self.dataset.data)
        plt.title('Yearly Violin Plot of Daily Average OLR')
        plt.xlabel('Year')
        plt.ylabel('Daily Average OLR')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset.base_path, f"olr_monthly_trend.png"), bbox_inches='tight')
        plt.close()

    def plot_daily_trend(self):
        plt.figure(figsize=(12, 6))
        sns.violinplot(x=self.dataset.data.index.day, y='OLR_mean', data=self.dataset.data)
        plt.title('Yearly Violin Plot of Daily Average OLR')
        plt.xlabel('Year')
        plt.ylabel('Daily Average OLR')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset.base_path, f"olr_daily_trend.png"), bbox_inches='tight')
        plt.close()

    def plot_overall_trend(self, plot_type='violin'):
        # Create a subplot layout
        _, ax = plt.subplots(figsize=(12, 6), dpi=300)
        
        df = self.dataset.data

        if plot_type == "violin":
            # Create formatting inputs
            xlabel = 'Year'

            # Add 'Year' and 'Month'
            df['Year'] = df.index.year
            df['Month'] = df.index.month_name().str[:3]

            # Violin Plot with Colors: visualises the distribution of data values for each spring month across years
            sns.violinplot(x='Year', y='OLR_mean', hue='Month', data=df, ax=ax, palette="muted", split=False)

            # Strip Plot: adds individual data points to the violin plot for detailed data visualization
            sns.stripplot(x='Year', y='OLR_mean', hue='Month', data=df, ax=ax, palette='dark:k', size=3, jitter=False, dodge=True)

            # Add grey box for visual separation of every other year for enhanced readability
            ax = self.add_grey_box(ax, df, plot_type)

            # Handling the legend to ensure clarity in distinguishing between different months
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:3], labels[:3], title='Month')
        elif plot_type == "line":
            # Create formatting inputs
            xlabel = 'Date'
            
            df = DataPlotter.make_date_axis_continuous(df)
            df['Year'] = df.index.year
            df['Year-Month'] = df.index.strftime('%Y-%m')

            # Drop non-numeric columns before resampling
            numeric_df = df.select_dtypes(include=[np.number])
            # Resample and aggregate only numeric columns
            weekly_mean_df = numeric_df.resample('W').agg(['min', 'mean', 'max'])
            monthly_mean_df = numeric_df.resample('M').agg(['min', 'mean', 'max'])
            # Fix time axis
            weekly_mean_df = DataPlotter.make_date_axis_continuous(weekly_mean_df)
            monthly_mean_df = DataPlotter.make_date_axis_continuous(monthly_mean_df, number_of_days=15)
            # Drop NaNs (creates gaps between years to avoid unclear datarepresentation)
            weekly_mean_df.dropna(inplace=False)
            monthly_mean_df.dropna(inplace=False)

            # Track legend entries
            legend_entries = []

            # Create colours 
            palette = sns.color_palette(n_colors=2)
    
            # Scatter plot for daily measurements
            ax.scatter(df['Year-Month-Day'], df['OLR_mean'], label=f'Daily OLR_mean', marker='.', s=1, color='black', alpha=0.75)

            # Line plot for weekly mean
            # weekly_color = np.clip(np.array(color) * 0.9, 0, 1)  # Darken the color
            weekly_line = ax.plot(weekly_mean_df['Year-Month-Day'], weekly_mean_df['OLR_mean']['mean'], label=f'Weekly Mean OLR', ls='-', lw=1, color=palette[0])
            
            # Line plot for monthly mean
            # monthly_color = np.clip(np.array(color) * 0.8, 0, 1)  # Darken the color further
            monthly_line = ax.plot(monthly_mean_df['Year-Month-Day'], monthly_mean_df['OLR_mean']['mean'], label=f'Monthly Mean OLR', ls='-', lw=2, marker='o', markersize=4, color=palette[1])
            
            # Add legend entry for this column
            legend_entries.append((weekly_line, monthly_line))

            ax.set_xticks(self.dataset.data['Year'].unique())
            # Add grey box for visual separation of every other year for enhanced readability
            self.add_grey_box(ax, self.dataset.data, plot_type)

        # Customizing the plot with titles and labels
        ax.set_title(f"MAM Average OLR at Nadir")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"OLR (mW m$^{{-2}})$")
        # ax.set_ylim([0.16, 0.3])
        ax.grid(axis='y', linestyle=':', color='k')
        ax.tick_params(axis='both', labelsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset.base_path, f"olr_trend_{plot_type}.png"), bbox_inches='tight', dpi=300)
        plt.close()
        



def main():
    # base_path = "G://My Drive//Research//Postdoc_2_CNRS_LATMOS//data//machine_learning//"
    base_path = "G:\\My Drive\\Research\\Postdoc_2_CNRS_LATMOS\\data\\iasi\\binned_olr\\"
    start_date = "2018-03-01"
    end_date = "2023-05-31"
    dataset = Dataset(base_path, start_date, end_date)
    plotter = DataPlotter(dataset)

    # plotter.plot_yearly_trend()
    # plotter.plot_monthly_trend()
    # plotter.plot_daily_trend()
    plotter.plot_overall_trend()

if __name__ == "__main__":
    main()
