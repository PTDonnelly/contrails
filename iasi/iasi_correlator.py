import pandas as pd

class IASICorrelator:
    def __init__(self, config: object):
        """
        Initialize the IASI correlator class with given parameters.

        Args:
        """
        self.config = config
        self.datafile_l1c = None
        self.datafile_l2 = None
        self.df_l1C = None
        self.df_l2 = None

    # def __enter__(self) -> 'IASICorrelator':
    #     """
    #     Opens the human-readable IASI data and prepares the correlation od spectra and ice clouds.
        
    #     Returns:
    #     self: The IASICorrelator object itself.
    #     """
    #     # Load the data
    #     self.load_data()
    #     return self

    def load_data(self):
        self.df_l1C = pd.read_csv(self.datafile_l1c)
        self.df_l2 = pd.read_csv(self.datafile_l2)

    def filter_spectra(self, datafile_l1c: str, datafile_l2: str):

        # Load the data
        self.load_data(datafile_l1c, datafile_l2)

        # Round latitudes, longitudes, and datetimes to two decimal places and truncate datetime to remove fractional second part
        df_L2[['Latitude', 'Longitude']] = df_L2[['Latitude', 'Longitude']].round(2)
        df_L1C[['Latitude', 'Longitude']] = df_L1C[['Latitude', 'Longitude']].round(2)
        df_L2['Datetime'] = df_L2['Datetime'].apply(lambda x: x.split('.')[0])
        df_L1C['Datetime'] = df_L1C['Datetime'].apply(lambda x: x.split('.')[0])

        # Merge dataframes on latitude, longitude and datetime
        merged_df = pd.merge(df_L2, df_L1C, on=['Latitude', 'Longitude', 'Datetime'])

        # Save the merged data
        merged_df.to_csv('merged_data.csv', index=False)
        return