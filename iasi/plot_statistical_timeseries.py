import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr
from datetime import datetime, timedelta
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose

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
        self.data.rename(columns={'OLR_mean': 'OLR'}, inplace=True)
        return

    @staticmethod
    def robust_z_score(df):
        # Calculate median and IQR
        median = df['OLR'].median()
        iqr = df['OLR'].quantile(0.75) - df['OLR'].quantile(0.25)
        
        # Calculate the robust Z-score
        df['Z-score'] = (df['OLR'] - median) / iqr
        return df

    @staticmethod
    def normalize_fractional_time(df, time_unit):
        """
        Normalize the time to be between March 1 and May 31, expressed as a fraction of the year.
        Args:
        df (DataFrame): Data with 'Year' and either 'Week' or 'Month'.
        time_unit (str): 'week' or 'month' to specify the unit of time aggregation.
        """
        # Constants for day of year for March 1 and May 31
        march_1_doy = pd.Timestamp(year=2020, month=3, day=1).dayofyear - 1
        may_31_doy = pd.Timestamp(year=2020, month=5, day=31).dayofyear - 1

        if time_unit == 'Day':
            # Convert date to day of year
            df['approx_day'] = df['Day']
        elif time_unit == 'Week':
            # Approximate day of year from week number, considering March 1 as start of week 9
            df['approx_day'] = (df['Week'] - 9) * 7 + march_1_doy
        elif time_unit == 'Month':
            # Approximate day by assigning mid-point to each month
            month_to_midpoint_doy = {
                3: pd.Timestamp(year=2020, month=3, day=15).dayofyear,
                4: pd.Timestamp(year=2020, month=4, day=15).dayofyear,
                5: pd.Timestamp(year=2020, month=5, day=15).dayofyear
            }
            df['approx_day'] = df['Month'].map(month_to_midpoint_doy)
        elif time_unit == 'Year':
            df['Date_continuous'] = df['Year'] + 0.5
            return df
        
        # Normalize the day of year to a fraction between 0 and 1
        df['Date_continuous'] = df['Year'] + ((df['approx_day'] - march_1_doy) / (may_31_doy - march_1_doy))

        return df
     
    @staticmethod
    def group_and_aggregate(df, group):
        # Group along time domain and aggregate OLR
        df = df.groupby(group)[['OLR', 'Z-score']].agg(['median'])
        
        # Reset index to format group labels as regular columns
        df = df.reset_index()

        # Apply continuous date transformation for plotting
        if len(group) > 1:
            df = Dataset.normalize_fractional_time(df, group[1])
        else:
            df = Dataset.normalize_fractional_time(df, group[0])
        
        # Drop NaNs
        df.dropna(inplace=True)

        # Rename OLR column back to OLR
        df.rename(columns={'mean': 'OLR'}, inplace=True)
        return df
    
    @staticmethod
    def resample_data(df):
        """
        Resamples data to generate straigtforward weekly, monthly, and yearly averages.
        
        Parameters:
        - df (pd.DataFrame): DataFrame with a datetime index.
        
        Returns:
        - Tuple containing DataFrames for weekly and monthly resampled data.
        """
        # Extracting year, month, and week number directly from the datetime index
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Week'] = df.index.isocalendar().week
        df['Day'] = df.index.dayofyear

        # Calculate Z-score for all daily OLR measurements with respect to the long-term median and IQR
        df = Dataset.robust_z_score(df)

        # Calculate the z-score for all daily measurements with respect to the long-term average
        daily_df = Dataset.group_and_aggregate(df, group=['Year', 'Day'])
        weekly_df = Dataset.group_and_aggregate(df, group=['Year', 'Week'])
        monthly_df = Dataset.group_and_aggregate(df, group=['Year', 'Month'])
        yearly_df = Dataset.group_and_aggregate(df, group=['Year'])

        return daily_df, weekly_df, monthly_df, yearly_df
    
    @staticmethod
    def restructure_data_weekdays(df):
        """
        Restructures data to generate more complicated temporal averages.
        
        Parameters:
        - df (pd.DataFrame): DataFrame with a datetime index.
        
        Returns:
        - Tuple containing DataFrames for weekly and monthly resampled data.
        """
        df = df.reset_index()
        # Ensure 'Date' is a datetime type if not already
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Weekday'] = df['Date'].dt.weekday

        # Calculate median and IQR for each Month-Weekday bin across all years
        grouped_stats = df.groupby(['Month', 'Weekday'])['OLR'].agg(['median', lambda x: x.quantile(0.75) - x.quantile(0.25)]).rename(columns={'<lambda_0>': 'IQR'}).reset_index()

        # Merge the statistical data back to the original DataFrame to compute Z-scores
        df = df.merge(grouped_stats, on=['Month', 'Weekday'])

        # Calculate robust Z-scores
        df['Z-score'] = (df['OLR'] - df['median']) / df['IQR']

        # Group by Year, Month, and Weekday and calculate the mean or median OLR
        grouped = df.groupby(['Year', 'Month', 'Weekday'])[['OLR', 'Z-score']].agg(['median']).reset_index()

        return grouped
    
    @staticmethod
    def restructure_data_weeks_in_month(df):
        """
        Restructures data to generate more complicated temporal averages.
        
        Parameters:
        - df (pd.DataFrame): DataFrame with a datetime index.
        
        Returns:
        - Tuple containing DataFrames for weekly and monthly resampled data.
        """
        df = df.reset_index()
        # Ensure 'Date' is a datetime type if not already
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        # df['Week'] = df['Date'].dt.isocalendar().week

        df['Day'] = df['Date'].dt.day
        df['Weekday'] = df['Date'].dt.weekday
        df['Week'] = df.groupby(['Year', 'Month'])['Day'].transform(lambda x: ((x - 1) // 7) + 1)

        # Calculate median and IQR for each Month-Week bin across all years
        grouped_stats = df.groupby(['Month'])['OLR'].agg(['median', lambda x: x.quantile(0.75) - x.quantile(0.25)]).rename(columns={'<lambda_0>': 'IQR'}).reset_index()

        # Merge the statistical data back to the original DataFrame to compute Z-scores
        df = df.merge(grouped_stats, on=['Month'])

        # Calculate robust Z-scores
        df['Z-score'] = (df['OLR'] - df['median']) / df['IQR']

        # Group by Year, Month, and Week and calculate the mean or median OLR
        grouped = df.groupby(['Year', 'Month', 'Week'])[['OLR', 'Z-score']].agg(['median']).reset_index()

        return grouped
    
    @staticmethod
    def restructure_data_months_in_year(df):
        """
        Restructures data to generate more complicated temporal averages.
        
        Parameters:
        - df (pd.DataFrame): DataFrame with a datetime index.
        
        Returns:
        - Tuple containing DataFrames for weekly and monthly resampled data.
        """
        df = df.reset_index()
        # Ensure 'Date' is a datetime type if not already
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month

        # Calculate median and IQR for each Month-Week bin across all years
        grouped_stats = df.groupby(['Month'])['OLR'].agg(['median', lambda x: x.quantile(0.75) - x.quantile(0.25)]).rename(columns={'<lambda_0>': 'IQR'}).reset_index()

        # Merge the statistical data back to the original DataFrame to compute Z-scores
        df = df.merge(grouped_stats, on=['Month'])

        # Calculate robust Z-scores
        df['Z-score'] = (df['OLR'] - df['median']) / df['IQR']

        # Group by Year, Month, and Week and calculate the mean or median OLR
        grouped = df.groupby(['Year', 'Month'])[['OLR', 'Z-score']].agg(['median']).reset_index()

        return grouped
    
    

class DataPlotter:
    def __init__(self, dataset):
        self.dataset = dataset
        self.df = self.dataset.data
    
    def add_grey_box(self, ax, unique_years, plot_type):
        """
        Adds grey boxes to the plot for every other year.

        Parameters:
        - ax (matplotlib.axes.Axes): The axes object to add the grey boxes to.
        - df (pd.DataFrame): DataFrame with 'Year' column.
        """
        for i, year in enumerate(unique_years):
            if plot_type == 'violin':
                if i % 2 == 0:
                    ax.axvspan(i-0.5, i+0.5, color='grey', alpha=0.2, zorder=0)
            elif plot_type == 'line':
                if i % 2 == 0:
                    ax.axvspan(year, year+1, color='grey', alpha=0.2, zorder=0)
        return ax

    def plot_overall_trend(self, plot_type='absolute'):
        # Plotting parameters
        _, axes = plt.subplots(3, 1, figsize=(9, 9), dpi=300, sharex=True)
        unique_years = sorted(self.df.index.year.unique())
        palette = sns.color_palette(n_colors=3)

        # Get data
        daily_df, weekly_df, monthly_df, yearly_df = Dataset.resample_data(self.df)

        # Plot each trend in a separate subplot
        axes[0].scatter(daily_df['Date_continuous'], daily_df['OLR'], label='Daily', marker='.', s=2, color='black', alpha=0.75)
        axes[0].set_title("IASI Integrated Radiances: MAM")

        axes[0].plot(weekly_df['Date_continuous'], weekly_df['OLR'], label='Weekly Mean', ls='-', lw=2, marker='o', markersize=4, color=palette[0])
        # axes[0].set_xlabel('Year')
        axes[0].set_ylabel(r"IIR $mW m^{-2}$")

        axes[1].plot(monthly_df['Date_continuous'], monthly_df['OLR'], label='Monthly Mean', ls='-', lw=2, marker='o', markersize=4, color=palette[1])
        # axes[1].set_xlabel('Month')
        axes[1].set_ylabel(r"IIR $mW m^{-2}$")

        axes[2].plot(yearly_df['Date_continuous'], yearly_df['OLR'], label='Yearly Mean', ls='-', lw=2, marker='o', markersize=4, color=palette[2])
        axes[2].set_xlabel('Year')
        axes[2].set_ylabel(r"IIR $mW m^{-2}$")

        # Grey boxes, legends, and grid for each axis
        for ax in axes:
            self.add_grey_box(ax, unique_years, plot_type='line')
            ax.legend(loc='lower right')
            ax.grid(axis='y', linestyle=':', color='k')
            ax.tick_params(axis='both', labelsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset.base_path, f"olr_trend_{plot_type}.png"), bbox_inches='tight', dpi=300)
        plt.close()

    def plot_temporal_trends_weekdays(self, plot_type='anomaly'):
        # Plotting parameters
        _, axes = plt.subplots(3, 1, figsize=(9, 9), dpi=300)
        unique_years = sorted(self.df.index.year.unique())
        palette = sns.color_palette(n_colors=len(unique_years))
        axes = axes.flatten()
        weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        months = ['March', 'April', 'May']

        if plot_type == 'absolute':
            quantity = 'OLR'
        elif plot_type == 'anomaly':
            quantity = 'Z-score'

        # Get data
        grouped = Dataset.restructure_data_weekdays(self.df)
        
        # Plot each month in a subplot
        for imonth, (month, ax) in enumerate(zip(months, axes)):
            data = grouped[grouped['Month'] == imonth + 3]
            for iyear, year in enumerate(data['Year'].unique()):
                ax.plot(data[data['Year'] == year]['Weekday'], data[data['Year'] == year][quantity]['median'], ls='-', lw=2, marker='o', markersize=4, color=palette[iyear], label=str(year))
            ax.set_title(month)
            ax.set_xticks(range(7))
            ax.set_xticklabels(weekdays)
            ax.legend(title='Year', loc='right')
            ax.grid(axis='y', linestyle=':', color='k')
            ax.tick_params(axis='both', labelsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset.base_path, f"olr_trend_weekdays_{plot_type}.png"), bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_temporal_trends_weeks_in_month(self, plot_type='absolute'):
        # Plotting parameters
        _, axes = plt.subplots(3, 1, figsize=(9, 9), dpi=300)
        unique_years = sorted(self.df.index.year.unique())
        palette = sns.color_palette(n_colors=len(unique_years))
        axes = axes.flatten()
        days = [1, 2, 3, 4, 5]
        months = ['March', 'April', 'May']

        if plot_type == 'absolute':
            quantity = 'OLR'
        elif plot_type == 'anomaly':
            quantity = 'Z-score'

        # Get data
        grouped = Dataset.restructure_data_weeks_in_month(self.df)
        
        # Plot each month in a subplot
        for imonth, (month, ax) in enumerate(zip(months, axes)):
            data = grouped[grouped['Month'] == imonth + 3]
            for iyear, year in enumerate(data['Year'].unique()):
                ax.plot(data[data['Year'] == year]['Week'], data[data['Year'] == year][quantity]['median'], ls='-', lw=2, marker='o', markersize=4, color=palette[iyear], label=str(year))
            ax.set_title(month)
            ax.set_xticks(days)
            ax.set_xticklabels(days)
            ax.legend(title='Year', loc='right')
            ax.grid(axis='y', linestyle=':', color='k')
            ax.tick_params(axis='both', labelsize=10)
            if imonth == 2:
                ax.set_xlabel('Week Number')

        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset.base_path, f"olr_trend_weeks_in_month_{plot_type}.png"), bbox_inches='tight', dpi=300)
        plt.close()

    def plot_temporal_trends_months_in_year(self, plot_type='absolute'):
        # Plotting parameters
        _, axes = plt.subplots(1, 1, figsize=(9, 3), dpi=300)
        unique_years = sorted(self.df.index.year.unique())
        palette = sns.color_palette(n_colors=len(unique_years))
        months = ['March', 'April', 'May']

        if plot_type == 'absolute':
            quantity = 'OLR'
        elif plot_type == 'anomaly':
            quantity = 'Z-score'

        # Get data
        grouped = Dataset.restructure_data_months_in_year(self.df)

        # Plot each month in a subplot
        for iyear, year in enumerate(grouped['Year'].unique()):
            data = grouped[grouped['Year'] == year]
            print(data.head())
            axes.plot(data[data['Year'] == year]['Month'], data[data['Year'] == year][quantity]['median'], ls='-', lw=2, marker='o', markersize=4, color=palette[iyear], label=str(year))
            axes.set_title('IIR')
            axes.set_xticks([3, 4, 5])
            axes.set_xticklabels(months)
            axes.legend(title='Year', loc='right')
            axes.grid(axis='y', linestyle=':', color='k')
            axes.tick_params(axis='both', labelsize=10)
            axes.set_xlabel('Month')

        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset.base_path, f"olr_trend_months_in_year_{plot_type}.png"), bbox_inches='tight', dpi=300)
        plt.close()


def main():
    # base_path = "G://My Drive//Research//Postdoc_2_CNRS_LATMOS//data//machine_learning//"
    base_path = "G:\\My Drive\\Research\\Postdoc_2_CNRS_LATMOS\\data\\iasi\\binned_olr\\"
    start_date = "2018-03-01"
    end_date = "2023-05-31"
    dataset = Dataset(base_path, start_date, end_date)
    plotter = DataPlotter(dataset)

    # plotter.plot_overall_trend()
    # plotter.plot_temporal_trends_weekdays()
    plotter.plot_temporal_trends_weeks_in_month() 
    plotter.plot_temporal_trends_months_in_year() 
    

if __name__ == "__main__":
    main()
