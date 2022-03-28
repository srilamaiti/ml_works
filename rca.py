import os
import numpy as np
import pandas as pd
import itertools
import datetime
import matplotlib.pyplot as plt
from numpy.core.shape_base import block
import logging
import sys
from datetime import datetime

class rca:

    # We are going to use threshold to select values from our fluctuation summary file which are above threshold value
    threshold = 35
    bin_ranges = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    bin_labels = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def __init__(self):
        """rca constructor"""
        logging.basicConfig(filename=sys.argv[0]+ '.log', level=logging.INFO)
    
    def read_and_process_file(self, prod_category, file_name, level):
        
        """This function reads and processes the input file for a specific level. The level can be basecode or customer. In the process it generates a file in the format of 
        prod_category_<<prod_category>>_rca_procssed_file_<<level>>_level_<<startyear,2digit week>>_<<endyear,2digit week>>.csv"""
        
        print("Execution starteded for read_and_process_file")
        # Assigning to class variables
        self.prod_category = prod_category
        self.file_name_prefix = 'prod_category_' + str(self.prod_category)
        self.level = level
        self.executive_summary_file_name = self.file_name_prefix + '_' + self.level + '_level_executive_summary.txt'
        self.executive_summary_file_text = ''
        
        # Reading the input file
        self.df = pd.read_csv(file_name)
        
        # Formatting the column names
        self.df.columns = [c.lower().replace('[','_').replace(' ','_').replace(']','').replace('.','_').replace('[','').replace('__','_') for c in self.df.columns]
        
        # Creating the date range from min and max dates, which will be used later to format the file name
        self.original_process_date_range = self.df.time_week.min().replace('-','_') + '_' + self.df.time_week.max().replace('-','_')
        
        # Creating date range list
        self.date_list = list(self.df.time_week.drop_duplicates())
        
        # Finding out last 14 weeks dates
        self.date_list_last_14_weeks = self.date_list[-14:]
        self.process_date_range = min(self.date_list_last_14_weeks).replace('-','_') + '_' + max(self.date_list_last_14_weeks).replace('-','_')
        
        # Retaining original data
        self.orig_df = self.df.copy()

        # Taking last 14 weeks of data
        self.df = self.df[self.df.time_week.isin(self.date_list_last_14_weeks)]
        
        # Handling NULL values with '-9999'
        self.df = self.df.fillna('-9999')

        #print(1)        
        # Based on the level supplied, we are extracting master data information.
        # If it's basecode level, then we are extracting basecode and it's decsription along with product category.
        # If it's customer level, then we are extracting IBP customer and it's decsription.
        if self.level == 'basecode':
            
            self.basecode_df = self.orig_df[['item_product_category', 'item_base_code', 'item_planning_item_description']].drop_duplicates()
            self.basecode_df.columns = ['prod_category', 'basecode', 'basecode_desc']
            self.basecode_df['basecode'] = self.basecode_df['basecode'].astype(str)
            self.df.rename({'ml_salesorg_accuracy' : 'ml_accuracy', 'item_base_code' : 'basecode'}, axis = 1, inplace=True)
            
        elif self.level == 'sales_domain_ibp_customer':
            
            self.cust_df = self.orig_df[['sales_domain_ibp_customer', 'sales_domain_channel_group_description']].drop_duplicates()
            self.cust_df.columns = ['ibp_customer', 'ibp_customer_desc']
        
        # Removing '%' from ml_accuracy and converting to float
        self.df['ml_accuracy'] = self.df.apply(lambda x : x.ml_accuracy.split('%')[0], axis = 1)
        self.df['ml_accuracy'] = self.df['ml_accuracy'].astype(float)
        #print(2)  

        # Removing ',' from actual_shifted and converting to float
        self.df['actual_shifted'] = self.df.apply(lambda x : x.actual_shifted.replace(',', ''), axis = 1)
        self.df['actual_shifted'] = self.df['actual_shifted'].astype(float)
        #print(3)          

        # Removing ',' from ml_fcst_lag2 and converting to float
        self.df['ml_fcst_lag2'] = self.df.apply(lambda x : x.ml_fcst_lag2.replace(',', ''), axis = 1)
        self.df['ml_fcst_lag2'] = self.df['ml_fcst_lag2'].astype(float)
        #print(4)  

        # Bringing back last week's data in the same row, so that we can do a comparison
        if self.level == 'basecode':
            self.df_shifted = self.df.groupby(['basecode'])[['time_week', 'actual_shifted', 'ml_fcst_lag2', 'ml_accuracy']].shift(fill_value=0)
        elif self.level == 'sales_domain_ibp_customer':
            self.df_shifted = self.df.groupby(['sales_domain_ibp_customer'])[['time_week', 'actual_shifted', 'ml_fcst_lag2', 'ml_accuracy']].shift(fill_value=0)
        
        self.df_shifted.columns = ['lag_1_wk_' + c for c in self.df_shifted.columns]
        self.df = pd.concat([self.df, self.df_shifted], axis=1)
        #print(5)  
        
        # Getting abs delta between ml_accuracy and lag_1_wk_ml_accuracy
        self.df['abs_delta_ml_accuracy_week_by_week'] = np.abs(self.df['ml_accuracy'].replace(-9999.0,0) - self.df['lag_1_wk_ml_accuracy'].replace(-9999.0,0))
        #print(6)  

        # Getting abs delta between ml_accuracy and lag_1_wk_ml_accuracy
        self.df['delta_ml_accuracy_week_by_week'] = self.df['ml_accuracy'].replace(-9999.0,0) - self.df['lag_1_wk_ml_accuracy'].replace(-9999.0,0)
        #print(7)

        #self.df['actual_polulated_flag'] = np.where(self.df['actual_shifted'].isna(), True, False)
        
        # Formulating conditions
        conditions = [
                      ((np.round(self.df['actual_shifted'], 0) == -9999) & (np.round(self.df['ml_fcst_lag2']) > -9999)),
                      ((np.round(self.df['actual_shifted'], 0) > -9999) & (np.round(self.df['ml_accuracy'], 0) == 0))
                     ]
        
        # Devising the condition messages
        values = ['actual_shifted_not_populated_ml_fsct_populated', 'actual_shifted_populated_and_ml_accuracy_0']
        
        # Assigning condition specific messages to each data point
        self.df['proc_msg'] = np.select(conditions, values)    
        
        # Writing the processed dataset in a file, this file will be the basis for next level analysis
        self.df.to_csv(self.file_name_prefix + '_rca_procssed_file_' + self.level + '_level_' + self.process_date_range + '.csv', index=False)
        print("File generated", self.file_name_prefix + '_rca_procssed_file_' + self.level + '_level_' + self.process_date_range + '.csv')

        # Dropping the description columns from the dataset.
        if self.level == 'basecode':
            self.df.drop(columns = ['item_planning_item_description', 'item_product_category'], inplace=True)
        elif self.level == 'sales_domain_ibp_customer':
            self.df.drop(columns = ['sales_domain_channel_group_description', 'item_product_category'], inplace=True)
        
        # Excluding records where there is no actual shifted populated
        self.df = self.df[np.round(self.df.actual_shifted,0) != -9999]
        
        # Excluding records where both ML and KHC forecasts are equally off
        #self.df = self.df[self.df['proc_msg'] != 'both_ml_khc_accuracy_are_off']
      
        print("Execution ended for read_and_process_file")
    
    def plot_abs_fluctuation_data(self, level):
        
        """This plotting function plots data based on derived column abs_delta_ml_accuracy_week_by_week"""
        print("Execution started for plot_abs_fluctuation_data")
        # Plotting the data for abs_delta_ml_accuracy_week_by_week per week basis
        self.df.pivot(index=['time_week'], columns=[level], values=['abs_delta_ml_accuracy_week_by_week']).fillna(0).plot(figsize = (20,10))
        plt.title("abs_delta_ml_accuracy_week_by_week at " + level)
        plt.show(block=False)
        print("Execution ended for plot_abs_fluctuation_data")

    def plot_fluctuation_data(self, level):
        
        """This plotting function plots data based on derived column delta_ml_accuracy_week_by_week"""
        print("Execution started for plot_fluctuation_data")
        # Plotting the data for delta_ml_accuracy_week_by_week per week basis
        self.df.pivot(index=['time_week'], columns=[level], values=['delta_ml_accuracy_week_by_week']).fillna(0).plot(figsize = (20,10))
        plt.title("delta_ml_accuracy_week_by_week at " + level)
        plt.show(block=False)
        print("Execution ended for plot_fluctuation_data")

    def plot_histogram_data(self, df, col_name):
        
        """This plotting function plots histogram data based on the col, refered as col_name in the input data frame df."""
        print("Execution started for plot_histogram_data")
        plt.hist(df[col_name])
        plt.title("Histogram for  at " + col_name)
        plt.show(block=False) 
        print("Execution ended for plot_histogram_data")

    def generate_bin_range_file(self, df, prod_category, kpi, level, nweek):
        """This function generates bin range file"""
        print(f"Execution started for generate_bin_range_file for prod_category {prod_category}, kpi {kpi}, level {level} over last {nweek} weeks")
        count_df = df.groupby(['bin_range']).agg({"bin_range" : "count"})
        count_df.columns = ['bin_count']
        count_df.reset_index(inplace=True)
        count_df.columns = ['fluctuation_pct_range','bin_count']
        file_name = 'prod_category_' + str(prod_category) + '_' + level + '_level_last_' + str(nweek) + '_' + kpi + '_fluctuation_stats.csv'
        count_df.to_csv(file_name, index = False)
        print("File generated", file_name)
        self.executive_summary_file_text += f"Fluctuation summary for product category: {prod_category}, kpi: {kpi}, level: {level} over last {nweek} weeks:-"
        self.executive_summary_file_text += "\n"
        self.executive_summary_file_text += count_df.to_string()
        self.executive_summary_file_text += "\n"
        self.executive_summary_file_text += "*" * 200
        self.executive_summary_file_text += "\n"
        print(f"Execution ended for generate_bin_range_file for prod_category {prod_category}, kpi {kpi}, level {level} over last {nweek} weeks")

    def generate_above_threshold_file(self, df, prod_category, kpi, level, nweek):
        """This function generates above threshold file"""
        print(f"Execution started for generate_above_threshold_file for prod_category {prod_category}, kpi {kpi}, level {level} over last {nweek} weeks")
        self.executive_summary_file_text += f"Above threshold ({self.threshold}%) records for product category: {prod_category}, kpi: {kpi}, level: {level} over last {nweek} weeks:-"
        self.executive_summary_file_text += "\n"
        self.executive_summary_file_text += df.to_string()
        self.executive_summary_file_text += "\n"
        self.executive_summary_file_text += "*" * 200
        self.executive_summary_file_text += "\n"
        print(f"Execution ended for generate_above_threshold_file for prod_category {prod_category}, kpi {kpi}, level {level} over last {nweek} weeks ")

    def generate_sparse_or_no_data_file(self, df, prod_category, kpi, level, nweek):
        """This function generates above sparse or no data file"""
        print(f"Execution started for generate_sparse_or_no_data_file for prod_category {prod_category}, kpi {kpi}, level {level} over last {nweek} weeks")
        self.executive_summary_file_text += f"Below for product category: {prod_category}, kpi: {kpi}, level: {level} over last {nweek} weeks either do not have consistent forecast or no forecast at all:-"
        self.executive_summary_file_text += "\n"
        self.executive_summary_file_text += df.to_string()
        self.executive_summary_file_text += "\n"
        self.executive_summary_file_text += "*" * 200
        self.executive_summary_file_text += "\n"
        print(f"Execution ended for generate_sparse_or_no_data_file for prod_category {prod_category}, kpi {kpi}, level {level} over last {nweek} weeks ")

    def analyze_data_fluctuation(self, kpi_val):
        
        """This function analyzes data based on abs_delta_ml_accuracy_week_by_week and generates a summary file for 4 weeks average fluctation and 8 week average fluctuation based on the level. 
        Those summary files are produced to the business users."""
        kpi = kpi_val # "abs_delta_ml_accuracy_week_by_week" / "delta_ml_accuracy_week_by_week"

        print("Execution started for analyze_data_fluctuation for kpi " + kpi)
        # Pivoting the data set, so that weeks come in rows and each basecode level/customer level kpi comes in differrent columns
        # Presenting the data this way makes it analyze easier.
        if self.level == 'basecode':
            self.analysis_df = self.df.pivot(index=['time_week'], columns=['basecode'], values=[kpi])
            self.analysis_df.columns = [c[0] + '_basecode_' + str(c[1]) for c in self.analysis_df.columns]
        elif self.level == 'sales_domain_ibp_customer':
            self.analysis_df = self.df.pivot(index=['time_week'], columns=['sales_domain_ibp_customer'], values=[kpi])
            self.analysis_df.columns = [c[0] + '_' + c[1].replace('-','_') for c in self.analysis_df.columns]
        
        # Keeping track of orifinal columns as we will use additional columns on top of this later on.
        orig_col_list = self.analysis_df.columns
        
        # Calculates max and min across basecode/customer level, week wise data using kpi
        max_kpi = 'max_' + kpi + '_in_week'
        min_kpi = 'min_' + kpi + '_in_week'
        zero_kpi = '0_' + kpi + '_in_week'
        
        max_kpi_over_the_period = 'max_' + kpi+ '_over_the_period'
        min_kpi_over_the_period = 'min_' + kpi+ '_over_the_period'
        avg_kpi_over_the_period = 'avg_' + kpi+ '_over_the_period'
        
        self.analysis_df[max_kpi] = self.analysis_df.max(axis=1)
        self.analysis_df[min_kpi] = self.analysis_df.min(axis=1)
        
        # Finding out missing kpi per week
        self.analysis_df['missing_accuracy_in_week'] = self.analysis_df.isnull().sum(axis = 1)
        
        # Trying to find out records per week where kpi = 0 for each basecode/customer level
        self.analysis_df[zero_kpi] = self.analysis_df.isin([0.00]).sum(axis=1)
        
        if self.level == 'basecode':
            
            # Finding out product where max diviation (max_kpi) is ocuurring at basecode level on weekly data
            # and assign it in max_diviation_basecode column
            prod_list = []
            for idx, row_value in self.analysis_df.iterrows():
                temp_df = pd.DataFrame(self.analysis_df.loc[idx])
                max_diviation_value = temp_df.loc[max_kpi][0]
                prod_list.append(temp_df[temp_df == temp_df.loc[max_kpi][0]].dropna().head(1).index[0][-21:])
            
            self.analysis_df['max_diviation_basecode']    = prod_list
            
        elif self.level == 'sales_domain_ibp_customer':
            
            # Finding out product where max diviation (max_kpi) is ocuurring at customer level on weekly data
            # and assign it in max_diviation_ibp_customer column
            cust_list = []
            for idx, row_value in self.analysis_df.iterrows():
                
                temp_df = pd.DataFrame(self.analysis_df.loc[idx])
                max_diviation_value = temp_df.loc[max_kpi][0]
                cust_list.append(temp_df[temp_df == temp_df.loc[max_kpi][0]].dropna().head(1).index[0][-11:])
            
            self.analysis_df['max_diviation_ibp_customer']    = cust_list
        
        # The analysis is wriiten back in 13 week details file
        file_name_part = '_level_weekly_abs_fluctuation_analysis_details_' if kpi[:3] == 'abs' else '_level_weekly_fluctuation_analysis_details_'
        self.analysis_df.to_csv(self.file_name_prefix + '_13week_' + self.level + file_name_part + self.process_date_range + '.csv')
        print("File generated", self.file_name_prefix + '_13week_' + self.level + file_name_part + self.process_date_range + '.csv')        

        # Now iterating over certain weeks of interest.
        # Here we are selecting 4 weeks and 8 weeks
        last_n_row_process_list = [4, 8]
        for last_n_row in last_n_row_process_list:
            
            # Subsetting the last n weeks of data based on week counts in interest
            analysis_subset_df = self.analysis_df.iloc[-last_n_row:]
            
            if self.level == 'basecode':
                
                last_n_weeks_basecode_diff_summary_list = []
                for c in orig_col_list:
                    
                    # Getting list of last n weeks basecode level data for abs_delta_ml_accuracy_week_by_week in a list and handle NULL values by -10000
                    last_nweek_basecode_level_diff_pct_list = analysis_subset_df[c].fillna(-10000).values.tolist()
                    
                    # Getting the length of the list of last n weeks basecode level data for abs_delta_ml_accuracy_week_by_week in a list and handle NULL values by -10000
                    init_length = len(last_nweek_basecode_level_diff_pct_list)
                    
                    # All records are nan, so the set will return -10000.0 and length of the set will be 1
                    # It's a special corner case where all the n weeks of interset does not have abs_delta_ml_accuracy_week_by_week
                    if len(set(last_nweek_basecode_level_diff_pct_list)) == 1:
                        last_n_weeks_basecode_diff_summary_list.append((c[-21:], np.nan, np.nan, np.nan, 'No forecast present'))
                    else:
                        
                        # keep removing -10000.0 from the list before doing any processing as we do not want the value to interfere with the min/max/avg value
                        while -10000.0 in last_nweek_basecode_level_diff_pct_list: 
                            last_nweek_basecode_level_diff_pct_list.remove(-10000.0)
                        
                        # Calculating max, min and avg on the non null values(abs_delta_ml_accuracy_week_by_week) of basecodes
                        processed_length = len(last_nweek_basecode_level_diff_pct_list)
                        last_n_weeks_basecode_diff_summary_list.append((c[-21:], max(last_nweek_basecode_level_diff_pct_list), min(last_nweek_basecode_level_diff_pct_list), sum(last_nweek_basecode_level_diff_pct_list) /  processed_length, ''))
                
                # Building basecode_level_last_nwk_analysis_agg_df based on the last n weeks agg data
                basecode_level_last_nwk_analysis_agg_df = pd.DataFrame(last_n_weeks_basecode_diff_summary_list, columns = ['basecode' , max_kpi_over_the_period, min_kpi_over_the_period, avg_kpi_over_the_period, 'message_1'])
                
                # If there is only one week's data present over last n weeks, have it marked.
                conditions = [np.round(basecode_level_last_nwk_analysis_agg_df[min_kpi_over_the_period], 0) == np.round(basecode_level_last_nwk_analysis_agg_df[max_kpi_over_the_period], 0)]
                values = ['only_one_value_present_over_the_period_considered']
                basecode_level_last_nwk_analysis_agg_df['message_2'] = np.select(conditions, values)   
                
                # Assigning processing period of last n weeks
                basecode_level_last_nwk_analysis_agg_df['processing_period'] = 'last ' + str(last_n_row) + ' weeks'
                
                # Reformatting back the column basecode, by removing basecode_
                basecode_level_last_nwk_analysis_agg_df['basecode'] = basecode_level_last_nwk_analysis_agg_df.apply(lambda x: x['basecode'].replace('basecode_',''), axis = 1)
                
                # Merging the data with master data to get back the basecode descriptions
                basecode_level_last_nwk_analysis_agg_df = pd.merge(self.basecode_df, basecode_level_last_nwk_analysis_agg_df, on = 'basecode', how = 'inner')

                # Adding bin_range to bucketize the fluctuations
                basecode_level_last_nwk_analysis_agg_df['bin_label'] = pd.cut(basecode_level_last_nwk_analysis_agg_df[avg_kpi_over_the_period], bins = self.bin_ranges, labels = self.bin_labels)
                basecode_level_last_nwk_analysis_agg_df['bin_range'] = pd.cut(basecode_level_last_nwk_analysis_agg_df[avg_kpi_over_the_period], bins = self.bin_ranges)
                self.plot_histogram_data(basecode_level_last_nwk_analysis_agg_df, avg_kpi_over_the_period)
                self.generate_bin_range_file(df = basecode_level_last_nwk_analysis_agg_df, prod_category = self.prod_category, kpi = kpi, level = self.level, nweek =  last_n_row)

                # Analysis at basecode level is written back to the file
                file_name_part_1 = '_wk_abs_fluctuation_analysis_agg_df' if kpi[:3] == 'abs' else '_wk_fluctuation_analysis_agg_df'
                basecode_level_last_nwk_analysis_agg_df.to_csv(self.file_name_prefix + '_basecode_level_last_' + str(last_n_row) + file_name_part_1 + '.csv', index=False)
                print("File generated", self.file_name_prefix + '_basecode_level_last_' + str(last_n_row) + file_name_part_1 + '.csv')
                
                # Average fluctualtion at basecode level over n weeks are being calculated and sorted and written back to a file 
                file_name_part_2 = '_basecode_summary_with_abs_avg_fluctuation_over_' if kpi[:3] == 'abs' else '_basecode_summary_with_avg_fluctuation_over_'
                summary_df = (basecode_level_last_nwk_analysis_agg_df[(basecode_level_last_nwk_analysis_agg_df.message_1 == '') & \
                                                                      (basecode_level_last_nwk_analysis_agg_df.message_2 != 'only_one_value_present_over_the_period_considered')\
                                                                     ].sort_values(by=[avg_kpi_over_the_period], ascending=False)\
                                                                     [['basecode', 'basecode_desc', 'bin_label', 'bin_range', avg_kpi_over_the_period]]\
                                                                     .reset_index()[['basecode', 'basecode_desc', 'bin_label', 'bin_range', avg_kpi_over_the_period]])                
                summary_df.to_csv(self.file_name_prefix + file_name_part_2 + str(last_n_row) +  '_weeks.csv')
                print("File generated", self.file_name_prefix + file_name_part_2 + str(last_n_row) +  '_weeks.csv')

                file_name_part_3 = '_top_basecode_avg_abs_fluctuation_summary_last_' if kpi[:3] == 'abs' else '_top_basecode_avg_fluctuation_summary_last_'
                above_threshold_summary_df = summary_df[summary_df[avg_kpi_over_the_period].apply(lambda x: round(x, 0)) >= self.threshold]
                above_threshold_summary_df.to_csv(self.file_name_prefix + file_name_part_3 + str(last_n_row) +  '_weeks.csv', index=False)
                print("File generated", self.file_name_prefix + file_name_part_3 + str(last_n_row) +  '_weeks.csv')
                if len(above_threshold_summary_df) > 0:
                    print("Calling generate_above_threshold_file")
                    self.generate_above_threshold_file(df = above_threshold_summary_df, prod_category = self.prod_category, kpi = kpi, level = self.level, nweek = last_n_row)
                else:
                    self.executive_summary_file_text += f"There is no record above threshold ({self.threshold}%) records for product category: {self.prod_category}, kpi: {kpi}, level: {self.level} over last {last_n_row} weeks"
                    self.executive_summary_file_text += "\n"
                    self.executive_summary_file_text += "*" * 200
                    self.executive_summary_file_text += "\n"

                # Writing down the records where for last n weeks there are sparse forecast or no foreacst at all
                file_name_part_4 = '_basecode_level_abs_delta_ml_accuracy_last_' if kpi[:3] == 'abs' else '_basecode_level_delta_ml_accuracy_last_'
                basecode_level_last_nwk_sparse_or_no_forecast = (basecode_level_last_nwk_analysis_agg_df[(basecode_level_last_nwk_analysis_agg_df.message_1 != '') |
                                                                                                        (basecode_level_last_nwk_analysis_agg_df.message_2 == 'only_one_value_present_over_the_period_considered')\
                                                                                                       ][['basecode', 'basecode_desc']])
                basecode_level_last_nwk_sparse_or_no_forecast.to_csv(self.file_name_prefix + file_name_part_4 + str(last_n_row) +  '_weeks_sparse_or_no_forecast.csv', index=False)
                print("File generated", self.file_name_prefix + file_name_part_4 + str(last_n_row) +  '_weeks_sparse_or_no_forecast.csv')
                if len(basecode_level_last_nwk_sparse_or_no_forecast) > 0:
                    print("Calling generate_sparse_or_no_data_file")
                    self.generate_sparse_or_no_data_file(df = basecode_level_last_nwk_sparse_or_no_forecast, prod_category = self.prod_category, kpi = kpi, level = self.level, nweek = last_n_row)

            elif self.level == 'sales_domain_ibp_customer':
                
                last_n_weeks_cust_diff_summary_list = []
                for c in orig_col_list:
                    
                    # Getting list of last n weeks customer level data for abs_delta_ml_accuracy_week_by_week in a list and handle NULL values by -10000
                    last_nweek_cust_level_diff_pct_list = analysis_subset_df[c].fillna(-10000).values.tolist()
                    
                    # Getting the length of the list of last n weeks basecode level data for abs_delta_ml_accuracy_week_by_week in a list and handle NULL values by -10000
                    init_length = len(last_nweek_cust_level_diff_pct_list)
                    
                    # All records are nan, so the set will return -10000.0 and length of the set will be 1
                    # It's a special corner case where all the n weeks of interset does not have abs_delta_ml_accuracy_week_by_week
                    if len(set(last_nweek_cust_level_diff_pct_list)) == 1:
                        
                        last_n_weeks_cust_diff_summary_list.append((c[-11:], np.nan, np.nan, np.nan, 'No forecast present'))
                        
                    else:
                        
                        # keep removing -10000.0 from the list before doing any processing as we do not want the value to interfere with the min/max/avg value
                        while -10000.0 in last_nweek_cust_level_diff_pct_list: 
                            last_nweek_cust_level_diff_pct_list.remove(-10000.0)
                        
                        # Calculating max, min and avg on the non null values(abs_delta_ml_accuracy_week_by_week) of customers
                        processed_length = len(last_nweek_cust_level_diff_pct_list)
                        last_n_weeks_cust_diff_summary_list.append((c[-11:], max(last_nweek_cust_level_diff_pct_list), min(last_nweek_cust_level_diff_pct_list), sum(last_nweek_cust_level_diff_pct_list) /  processed_length, ''))
                
                # Building cust_level_last_nwk_analysis_agg_df based on the last n weeks agg data
                cust_level_last_nwk_analysis_agg_df = pd.DataFrame(last_n_weeks_cust_diff_summary_list, columns = ['ibp_customer' , max_kpi_over_the_period, min_kpi_over_the_period, avg_kpi_over_the_period, 'message_1'])
                
                # If there is only one week's data present over last n weeks, have it marked.
                conditions = [np.round(cust_level_last_nwk_analysis_agg_df[min_kpi_over_the_period], 0) == np.round(cust_level_last_nwk_analysis_agg_df[max_kpi_over_the_period], 0)]
                values = ['only_one_value_present_over_the_period_considered']
                cust_level_last_nwk_analysis_agg_df['message_2'] = np.select(conditions, values)   
                
                # Assigning processing period of last n weeks
                cust_level_last_nwk_analysis_agg_df['processing_period'] = 'last ' + str(last_n_row) + ' weeks'
                
                # Reformatting back the column basecode, by replacing "_" with "-"
                cust_level_last_nwk_analysis_agg_df['ibp_customer'] = cust_level_last_nwk_analysis_agg_df.apply(lambda x: x['ibp_customer'].replace('_','-'), axis = 1)
                
                # Merging the data with master data to get back the IBP customer descriptions
                cust_level_last_nwk_analysis_agg_df = pd.merge(self.cust_df, cust_level_last_nwk_analysis_agg_df, on = 'ibp_customer', how = 'inner')

                # Adding bin_range to bucketize the fluctuations
                cust_level_last_nwk_analysis_agg_df['bin_label'] = pd.cut(cust_level_last_nwk_analysis_agg_df[avg_kpi_over_the_period], bins = self.bin_ranges, labels = self.bin_labels)
                cust_level_last_nwk_analysis_agg_df['bin_range'] = pd.cut(cust_level_last_nwk_analysis_agg_df[avg_kpi_over_the_period], bins = self.bin_ranges)
                self.plot_histogram_data(cust_level_last_nwk_analysis_agg_df, avg_kpi_over_the_period)
                self.generate_bin_range_file(df = cust_level_last_nwk_analysis_agg_df, prod_category = self.prod_category, kpi = kpi, level = self.level, nweek =  last_n_row)
                
                # Analysis at customer level is written back to the file
                file_name_part_1 = '_wk_abs_fluctuation_analysis_agg_df' if kpi[:3] == 'abs' else '_wk_fluctuation_analysis_agg_df'
                cust_level_last_nwk_analysis_agg_df.to_csv(self.file_name_prefix + '_cust_level_last_' + str(last_n_row) + file_name_part_1 + '.csv', index=False)
                print("File generated", self.file_name_prefix + '_cust_level_last_' + str(last_n_row) + file_name_part_1 + '.csv')
                
                # Average fluctualtion over n weeks at customer level are being calculated and sorted and written back to a file 
                file_name_part_2 = '_customer_summary_with_abs_avg_fluctuation_over_' if kpi[:3] == 'abs' else '_customer_summary_with_avg_fluctuation_over_'
                summary_df = cust_level_last_nwk_analysis_agg_df[(cust_level_last_nwk_analysis_agg_df.message_1 == '') & (cust_level_last_nwk_analysis_agg_df.message_2 != 'only_one_value_present_over_the_period_considered')].sort_values(by=[avg_kpi_over_the_period], ascending=False)[['ibp_customer', 'ibp_customer_desc', 'bin_label', 'bin_range', avg_kpi_over_the_period]].reset_index()[['ibp_customer', 'ibp_customer_desc', 'bin_label', 'bin_range', avg_kpi_over_the_period]]
                summary_df.to_csv(self.file_name_prefix + file_name_part_2 + str(last_n_row) +  '_weeks.csv')
                print("File generated", self.file_name_prefix + file_name_part_2 + str(last_n_row) +  '_weeks.csv')
                
                file_name_part_3 = '_top_customer_avg_abs_fluctuation_summary_last_' if kpi[:3] == 'abs' else '_top_customer_avg_fluctuation_summary_last_'                
                above_threshold_summary_df = summary_df[summary_df[avg_kpi_over_the_period].apply(lambda x: round(x, 0)) >= self.threshold]
                above_threshold_summary_df.to_csv(self.file_name_prefix + file_name_part_3 + str(last_n_row) +  '_weeks.csv', index=False)
                print("File generated", self.file_name_prefix + file_name_part_3 + str(last_n_row) +  '_weeks.csv')
                if len(above_threshold_summary_df) > 0:
                    print("Calling generate_above_threshold_file")
                    self.generate_above_threshold_file(df = above_threshold_summary_df, prod_category = self.prod_category, kpi = kpi, level = self.level, nweek = last_n_row)
                else:
                    self.executive_summary_file_text += f"There is no record above threshold ({self.threshold}%) records for product category: {self.prod_category}, kpi: {kpi}, level: {self.level} over last {last_n_row} weeks"
                    self.executive_summary_file_text += "\n"
                    self.executive_summary_file_text += "*" * 200
                    self.executive_summary_file_text += "\n"

                # Writing down the records where for last n weeks there are sparse forecast or no foreacst at all
                file_name_part_4 = '_customer_level_abs_delta_ml_accuracy_last_' if kpi[:3] == 'abs' else '_customer_level_delta_ml_accuracy_last_'                
                cust_level_last_nwk_sparse_or_no_forecast = cust_level_last_nwk_analysis_agg_df[(cust_level_last_nwk_analysis_agg_df.message_1 != '') | (cust_level_last_nwk_analysis_agg_df.message_2 == 'only_one_value_present_over_the_period_considered')][['ibp_customer', 'ibp_customer_desc']]
                cust_level_last_nwk_sparse_or_no_forecast.to_csv(self.file_name_prefix + file_name_part_4 + str(last_n_row) +  '_weeks_sparse_or_no_forecast.csv', index=False)
                print("File generated", self.file_name_prefix + file_name_part_4 + str(last_n_row) +  '_weeks_sparse_or_no_forecast.csv')
                if len(cust_level_last_nwk_sparse_or_no_forecast) > 0:
                    print("Calling generate_sparse_or_no_data_file")
                    self.generate_sparse_or_no_data_file(df = cust_level_last_nwk_sparse_or_no_forecast, prod_category = self.prod_category, kpi = kpi, level = self.level, nweek = last_n_row)
                
        print("Execution ended for analyze_data_fluctuation for kpi " + kpi)

    def fun(self):
        """Fun function"""
        print("fun")

    def zip_files(self):
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d%H%M%S")
        file_name = "rca_executive_summary_" + str(dt_string) + ".zip"
        os_cmd = "zip " + file_name + ".txt"
        os.system(os_cmd)
        print("File generated", file_name)

    def pipeline(self, prod_category, file_name, level):

        """This function is a controller function to run all the processes in sequence."""

        """
        Sample files generated by the method calls for one category-customer level:-

        File generated prod_category_004_rca_procssed_file_sales_domain_ibp_customer_level_2021_W48_2022_W09.csv

        File generated prod_category_004_13week_sales_domain_ibp_customer_level_weekly_abs_fluctuation_analysis_details_2021_W48_2022_W09.csv
        
        File generated prod_category_004_sales_domain_ibp_customer_level_last_4_abs_delta_ml_accuracy_week_by_week_fluctuation_stats.csv
        File generated prod_category_004_cust_level_last_4_wk_abs_fluctuation_analysis_agg_df.csv
        File generated prod_category_004_customer_summary_with_abs_avg_fluctuation_over_4_weeks.csv
        File generated prod_category_004_top_customer_avg_abs_fluctuation_summary_last_4_weeks.csv
        File generated prod_category_004_customer_level_abs_delta_ml_accuracy_last_4_weeks_sparse_or_no_forecast.csv
        
        File generated prod_category_004_sales_domain_ibp_customer_level_last_8_abs_delta_ml_accuracy_week_by_week_fluctuation_stats.csv
        File generated prod_category_004_cust_level_last_8_wk_abs_fluctuation_analysis_agg_df.csv
        File generated prod_category_004_customer_summary_with_abs_avg_fluctuation_over_8_weeks.csv
        File generated prod_category_004_top_customer_avg_abs_fluctuation_summary_last_8_weeks.csv
        File generated prod_category_004_customer_level_abs_delta_ml_accuracy_last_8_weeks_sparse_or_no_forecast.csv
        
        File generated prod_category_004_13week_sales_domain_ibp_customer_level_weekly_fluctuation_analysis_details_2021_W48_2022_W09.csv
        
        File generated prod_category_004_sales_domain_ibp_customer_level_last_4_delta_ml_accuracy_week_by_week_fluctuation_stats.csv
        File generated prod_category_004_cust_level_last_4_wk_fluctuation_analysis_agg_df.csv
        File generated prod_category_004_customer_summary_with_avg_fluctuation_over_4_weeks.csv
        File generated prod_category_004_top_customer_avg_fluctuation_summary_last_4_weeks.csv
        File generated prod_category_004_customer_level_delta_ml_accuracy_last_4_weeks_sparse_or_no_forecast.csv
        
        File generated prod_category_004_sales_domain_ibp_customer_level_last_8_delta_ml_accuracy_week_by_week_fluctuation_stats.csv
        File generated prod_category_004_cust_level_last_8_wk_fluctuation_analysis_agg_df.csv
        File generated prod_category_004_customer_summary_with_avg_fluctuation_over_8_weeks.csv
        File generated prod_category_004_top_customer_avg_fluctuation_summary_last_8_weeks.csv
        File generated prod_category_004_customer_level_delta_ml_accuracy_last_8_weeks_sparse_or_no_forecast.csv
        
        File generated prod_category_004_sales_domain_ibp_customer_level_executive_summary.txt
        
        File generated rca_004_sales_domain_ibp_customer_analysis.zip
        """        
        #Read the input file and process it to make it ready for further analysis
        # Generates a file in the format 
        # prod_category_<<3 digit prod_category>>_rca_procssed_file_<<level>>_<<start year 4 digit>>_W<<2 digit week>>_<<end year 4 digit>>_W<<2 digit end week>>.csv
        # Ex. prod_category_004_rca_procssed_file_sales_domain_ibp_customer_level_2021_W48_2022_W09.csv
        self.read_and_process_file(prod_category, file_name, level)
        
        # Plot the delta measure
        self.plot_fluctuation_data(level)

        # Plot the absolute delta measure
        self.plot_abs_fluctuation_data(level)

        print("Calling analyze_data_fluctuation for kpi abs_delta_ml_accuracy_week_by_week")
        self.analyze_data_fluctuation('abs_delta_ml_accuracy_week_by_week')

        print("Calling analyze_data_fluctuation for kpi delta_ml_accuracy_week_by_week")
        self.analyze_data_fluctuation('delta_ml_accuracy_week_by_week')

        self.exec_summ_file = open(self.executive_summary_file_name, "a")
        self.exec_summ_file.write(self.executive_summary_file_text)
        self.exec_summ_file.close()
        print("File generated", self.executive_summary_file_name)

        self.zip_files()


file_list = [
   '004_basecode.csv'
 , '004_customer.csv'
 , '016_basecode.csv'
 , '016_customer.csv'
 , '038_basecode.csv'
 , '038_customer.csv'
 , '056_basecode.csv'
 , '056_customer.csv'
 , '131_basecode.csv'
 , '131_customer.csv'
 , '233_basecode.csv'
 , '233_customer.csv'
]
ctg_ctry_list = []
for f in file_list:
    if '_basecode' in f:
        ctg_ctry_list.append(f.split("_")[0])
    else:
        ctg_ctry_list.append(f.split("_")[0])
print(ctg_ctry_list)
for f, c in zip(file_list, ctg_ctry_list):
    print(f"Processing file {f}")
    analysis = rca()
    if '_basecode' in f:
        analysis.pipeline(c, f, 'basecode')
    else:
        analysis.pipeline(c, f, 'sales_domain_ibp_customer')