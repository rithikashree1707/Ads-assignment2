#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 19:43:07 2023

@author: tamadaritikashree
"""

# Import the pandas library as pd, used for analysis
import pandas as pd
# Import the matplotlib.pyplot as plt, used to create plots and charts
import matplotlib.pyplot as plt
# Import the seaborn library for attractive drawing
import seaborn as sns
# Importing the 'stats' module from the 'scipy' library
import scipy.stats as stats
# Import the 'reduce' function from the 'functools' library
from functools import reduce



def read_data(Indicator_Name):
    """ Reads data from a CSV file, filters it based on the Indicator Name and 
        a list of countries, performs data manipulation, and returns 
        two DataFrames.
        Parameters:
            Indicator_Name (str): The name of the indicator for filtering the 
            data.
         Returns:
            d1_d: Filtered and processed DataFrame excluding unwanted columns.
           d1_f: Transposed, cleaned, and formatted DataFrame for further 
           analysis."""
    # Read data from CSV file into a DataFrame
    df = pd.read_csv("API_19_DS2_en_csv_v2_6183479.csv", skiprows=3)
    # List countries to filter data
    countries = ['Bangladesh', 'India',  'Pakistan', 'Sri Lanka', 
                 'United Kingdom']
    # Filter the DataFrame  using the Indicator Name and Country Names
    d1 = df[(df['Indicator Name'] == Indicator_Name)&(df['Country Name']
                                                      .isin(countries))]
    # Drop the data which is unwanted and reset the index
    d1_d = d1.drop(['Country Code', 'Indicator Name', 'Indicator Code', '1960',
                    '1961', '1962', '1963', '1964', '1965', '1966', '1967', 
                    '1968','1969', '1970', '1971', '1972', '1973', '1974', 
                    '1975', '1976', '1977', '1978', '1979', '1980', '1981', 
                    '1982', '1983', '1984', '1985', '1986','1987', '1988', 
                    '1989', '1990', '1991', '1992', '1993', '1994', '1995',
                    '1996', '1997', '1998', '1999', '2000', '2001', '2002', 
                    '2003', '2004', '2005', '2006', '2007', '2008', '2009', 
                    '2010', '2011', '2012', '2013', '2014', '2021', '2022', 
                    'Unnamed: 67'], axis=1).reset_index(drop=True)
    # Transpose the data
    d1_t = d1_d.transpose()
    # We set the first row of the DataFrame as the column names.
    d1_t.columns = d1_t.iloc[0]
    # Select rows from index 1 to the end using iloc function
    d1_t = d1_t.iloc[1:]
    # Convert the index of DataFrame d1_t to numeric type
    d1_t.index = pd.to_numeric(d1_t.index)
    # Create a new column 'Years' in d1_t and assign the index values to it
    d1_t['Years'] = d1_t.index
    # Resetting the index of d1_t and dropping the old index
    d1_f = d1_t.reset_index(drop=True)
    # Assuming d1_d and d1_f are two DataFrames obtained from some operations
    return d1_d,d1_f


def slice_data(df1):
    """ Slice the input DataFrame to retain only 'Country Name' and '2017' 
        columns.
        Parameters:
        df1 : Input DataFrame containing data for multiple years.
        Returns:
        pandas.DataFrame: DataFrame with only 'Country Name' and '2017' 
        columns."""
    # Select specific columns 'Country Name' and '2017' from df1
    df1 = df1[['Country Name', '2017']]
    # Return df1
    return df1


def m_data(*dataframes):
    """ Merge multiple DataFrames based on 'Country Name' using an outer join.
        Parameters:
        *dataframes: Variable number of DataFrames to merge.
        returns:
        DataFrame: Merged DataFrame containing data from all input 
        DataFrames."""
    # Make use of functools.reduce to merge iteratively DataFrames
    # 'Left' and 'right' DataFrames are inputs to the lambda function
    # Using the 'Country Name' as column, pd.merge() executes an outer join.
    m_data = reduce(lambda left, right: pd.merge(left, right, 
                                on='Country Name', how='outer'), dataframes)
    # Reset the index of the merged_data and drop the previous index
    m_data = m_data.reset_index(drop=True)
    # Return merged_data
    return m_data

   
def heatmap(df):
    """ Generate a heatmap to visualize the correlation matrix of numerical 
        columns in the given DataFrame. It selects numerical columns, 
        calculates the correlation matrix, and plots the heatmap using 
        seaborn library."""
    # Plot the figure
    plt.figure()
    # Select columns with numeric data types from 'df'
    numeric_df = df.select_dtypes(include='number')
    # Calculate the correlation matrix using the corr() function
    correlation_matrix = numeric_df.corr()
    # Plotting a heatmap to visualize the correlation matrix
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    # Plot the title as 'Correlation'
    plt.title('Correlation')
    # Rotate the x-axis tick labels to 90 degrees 
    plt.xticks(rotation = 90)
    # Rotate the y-axis tick labels to 0 degrees 
    plt.yticks(rotation = 0)
    # Set the visual style of seaborn plots to "whitegrid"
    sns.set_style("whitegrid")
    # Save the figure
    plt.savefig('heatmap.png')
    # Show the plot
    plt.show()
    

  
def lineplot(df):
        
    """Generate a line plot for specific countries' population over the years.
       Parameters:
       df: DataFrame containing population data for various countries over 
       time."""
    # Plot the figure
    plt.figure()
    # Plot the line graph with 'years' as x-axis and countries as y-axis
    # give the label for x,y axis and put marker = 'o'
    df.plot(x='Years', 
             y=['Bangladesh', 'India',  'Pakistan', 'Sri Lanka', 
                'United Kingdom'], 
                 kind='line', xlabel='Years', ylabel='Population', marker='o')
    # Set the legend location to 'upper right' with fontsize as '5'
    plt.legend(loc='upper right', fontsize='5', labelspacing=1)
    # Plot the title as 'Renewable energy consumption'
    plt.title('Renewable energy consumption')
    # Show the plot
    plt.show()
     

     
def barplot(df):
    """Generate a bar plot for specific countries' population over the years.
       Parameters:
       df: DataFrame containing population data for various countries over 
       time."""
    # Plot the figure
    plt.figure()
    # Plot the line graph with 'years' as x-axis and countries as y-axis
    # give the label for x,y axis
    df.plot(x='Years', 
            y=['Bangladesh', 'India',  'Pakistan', 'Sri Lanka', 
               'United Kingdom'], 
                kind='bar', xlabel='Years', ylabel='total(%)')
    # Set the legend location to 'upper right' with fontsize as '5'
    plt.legend(loc='upper right', fontsize='5', labelspacing=1)
    # Plot the title as 'Mortality rate, under-5'
    plt.title('Mortality rate, under-5')
    # Show the plot
    plt.show()
    

    
def boxplot(df, countries):
    """Generate a box plot for specific countries' population over the years.
       Parameters:
       df: DataFrame containing population data for various countries over 
       time."""
    # Plot the figure
    plt.figure() 
    # Plot a boxplot for the data of specific countries from df
    #'palette' defines the color palette for the boxplot, here 'Set3' is used
    sns.boxplot(data=df[countries], palette='Set3')
    # Plot the title as 'Population Growth'
    plt.title('Population Growth')
    # Rotate the x-axis tick labels to 45 degrees 
    plt.xticks(rotation=45) 
    # Set x-axis label as 'Countries'
    plt.xlabel('Countries')
    # Set y-axis label as 'Population Growth'
    plt.ylabel('Population Growth')
    # Adjusts the layout to prevent overlapping of subplots or axes labels.
    plt.tight_layout()  
    # Save the figure
    plt.savefig('boxplot.png')
    # Show the plot
    plt.show()
    


def skewkurtplot(df):
    """ Plot a histogram and calculate skewness and kurtosis of a DataFrame.
    Parameters:
    df : Input DataFrame containing numeric data."""
    
    # Convert DataFrame to numeric values, coercing errors to NaN
    df_numeric = pd.to_numeric(df, errors='coerce')
    # Remove rows with missing values from the DataFrame 'df_numeric'
    df_numeric = df_numeric.dropna()
    # Calculating the skewness of numeric columns in the DataFrame
    skewness = stats.skew(df_numeric)
    # Calculate the kurtosis of a DataFrame containing numeric data
    kurtosis = stats.kurtosis(df_numeric)
    # Printing the skewness value using an f-string to format the output
    print(f"Skewness: {skewness}")
    # Printing the Kurtosis value along with its calculated result
    print(f"Kurtosis: {kurtosis}") 
    # Plot the figure
    plt.figure()
    # Plotting a histogram for the DataFrame 'df'
    # set bins to 10, alpha to 0.7, color to green and edgecolor to black
    plt.hist(df, bins=10, alpha=0.7, color= 'green', edgecolor='black')
    # Display gridlines on the plot
    plt.grid(True)
    # Plot the title as 'India'
    plt.title('India')
    # Set x-axis label as "Cereal yield  from 2015-2020"
    plt.xlabel("Cereal yield  from 2015-2020")
    # Set y-axis label as 'Frequency'
    plt.ylabel('Frequency')
    # Show the plot
    plt.show()    
       

# Re and Re_t are two variables used to store the loaded data
Re , Re_t  = read_data('Renewable energy consumption (% of total final energy consumption)')
# Pg , Pg_t are two variables used to store the loaded data
Pg , Pg_t  = read_data('Population growth (annual %)')
# Cy , Cy_t are two variables used to store the loaded data
Cy , Cy_t  = read_data('Cereal yield (kg per hectare)')
# Mr , Mr_t are two variables used to store the loaded data
Mr , Mr_t = read_data('Mortality rate, under-5 (per 1,000 live births)')

# Renaming the '2017' column to 'Renewable energy consumption' in sliced data
Re_c = slice_data(Re).rename(columns={'2017': 'Renewable energy consumption'})
# Renaming the '2017' column to 'Population growth' in sliced data
Pg_c = slice_data(Pg).rename(columns={'2017': 'Population growth '})
# Renaming the '2017' column to 'Cereal yield' in sliced data
Cy_c = slice_data(Cy).rename(columns={'2017': 'Cereal yield'})
# Renaming the '2017' column to 'Mortality rate, under-5' in sliced data
Mr_c = slice_data(Mr).rename(columns={'2017': 'Mortality rate, under-5'})

# Merging four DF: Re_c, Pg_c, Cy_c, and Mr_c using the merge_data function
df2 = m_data(Re_c, Pg_c, Cy_c , Mr_c)
# Displaying descriptive statistics of the DataFrame using describe()
df2.describe()
# Descriptive statistics include count, mean, standard deviation, min, max
print(df2.describe())
# Display a heatmap using the DataFrame df2
heatmap(df2)
# Display a lineplot using DataFrame Re_t
lineplot(Re_t)
# Display a barplot using DataFrame Mr_t
barplot(Mr_t)
# List of countries
countries = ['Bangladesh', 'India',  'Pakistan', 'Sri Lanka', 'United Kingdom']
# Display a boxplot using DataFrame Pg_t for selected countries
boxplot(Pg_t, countries)
# Display a skewkurtplot using 'India' data from DataFrame Cy_t
skewkurtplot(Cy_t['India'])   

 

    
                   