# Databricks notebook source
# SparkSession was introduced in version Spark 2.0, It is an entry point to underlying Spark functionality in order to programmatically create Spark RDD, DataFrame, and DataSet. SparkSession’s object spark is the default variable available in spark-shell and it can be created programmatically using SparkSession builder pattern.

# importing pandas library : open-source data manipulation and analysis library for Python. It provides easy-to-use data structures and functions for efficiently manipulating large datasets. 


import pandas as pd
from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder \
    .appName("NEIC_Earthquakes") \
    .getOrCreate()

# Disable Arrow optimization
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")


# Loading the data from csv file to spark- dataframe

df = spark.read.format("csv").option("header", "true").load("/mnt/mmmdata-dest/tmp/parul/database.csv")
# displaying the dataframe
df.display()

# COMMAND ----------

df=df.na.fill("NA")
df.display()

# COMMAND ----------

# The to_date() function in PySpark SQL converts a string to a date format. Here , we have Date column in the String data type. To apply various functions on date column, we are converting it into Date data type.

#pyspark.sql.functions:  A collections of builtin functions available for DataFrame operations.

from pyspark.sql.functions import to_date
from pyspark.sql.functions import *
from pyspark.sql import functions as F

# converting the Date column to Date type
df = df.withColumn("Date", to_date(F.col("Date"), "MM/dd/yyyy"))
 
# printSchema() is used to print or display the schema of the DataFrame in the tree format along with column name and data type.
# In the below output, we can see that the Date column has been converted into date type.
df.printSchema()

# COMMAND ----------




df.select('Date').limit(10).display()




 # Extracting the day of  month and day of the week from the Date column
#dayofweek -> Extract the day of the week of a given date/timestamp as integer. Ranges from 1 for a Sunday through to 7 for a Saturday
df =df.withColumn('day_of_month', dayofmonth('Date'))
df =df.withColumn('day_of_week', dayofweek('Date').cast("int"))


df.display()



# COMMAND ----------

# MAGIC %md
# MAGIC Starting the analysis of the data with the reuiqred questions:
# MAGIC How does the Day of a Week affect the number of earthquakes?
# MAGIC

# COMMAND ----------


# matplotlib.pyplot is a collection of command style functions that make matplotlib work like MATLAB. Each pyplot function makes some change to a figure: e.g., creates a figure, creates a plotting area in a figure, plots some lines in a plotting area, decorates the plot with labels, etc.

import matplotlib.pyplot as plt
import numpy as np

# Aggregating the data by counting the number of earthquakes as per day of week.  .alias() function is used to give an alias/reference name to a column which we can refer further.
# groupBy() : This groups the DataFrame by the specified column (here: "day_of_week") and calculates the count of rows in each group.

earthquake_count_by_day = df.groupBy("day_of_week").agg(count("*").alias("EarthquakeCount"))


earthquake_count_by_day = earthquake_count_by_day.orderBy('EarthquakeCount', ascending=False)
earthquake_count_by_day.show()

# Collect the results to the driver for visualization
results = earthquake_count_by_day.collect()

days = [row["day_of_week"] for row in results]
counts = [row["EarthquakeCount"] for row in results]


x_pos = np.arange(len(days))

plt.bar(x_pos, counts, align='center')
plt.xticks(x_pos, days) #Sets the labels for x-axis
plt.xlabel("Day of the Week")
plt.ylabel("Number of Earthquakes")
plt.title("Number of Earthquakes by Day of the Week")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC From the above graph and the result obtained, we can see that the maximum no of earthquakes occurred on 4th and 7th day of the week i.e. Wednesday and Saturday.
# MAGIC (Ranges from 1 for a Sunday through to 7 for a Saturday).
# MAGIC Wednesday : 3431 earthquakes
# MAGIC Saturday  : 3433 earthquakes
# MAGIC
# MAGIC

# COMMAND ----------

# relation between Day of the month and Number of earthquakes that happened in a year

# In PySpark, the withColumn method is used to add a new column or replace an existing column in a DataFrame. Here, we are creating a column "Year" usimg Year function.
# Similary we are using day_of_month function and using Date column to extract the required fields

df = df.withColumn('year', year('Date')).withColumn('day_of_month', dayofmonth('Date'))

# Group by Year and Day of Month, count the number of earthquakes
result = df.groupBy('year', 'day_of_month').count().orderBy('year', 'day_of_month')
result.display()


# Convert the PySpark DataFrame to Pandas for plotting
result_pd = result.toPandas()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(result_pd['day_of_month'], result_pd['count'], label='Number of Earthquakes', marker='o')
plt.title('Relation between Day of the Month and Number of Earthquakes (Year-wise)')
plt.xlabel('Day of the Month')
plt.ylabel('Number of Earthquakes')
plt.legend()
plt.show()




# COMMAND ----------

# MAGIC %md
# MAGIC "From the above graph, we can conclude that the number of earthquakes are maximum during the 10th - 13th of any month.

# COMMAND ----------

# Calculating the average frequency of earthquakes in a month from the year 1965 to 2016
# Extracting the Year and Month from Date column with year and month functions respectively.

df = df.withColumn("Year", year("Date"))
df = df.withColumn("Month", month("Date"))

# filter() method is used to select rows from a DataFrame that satisfy a specified condition 
df_filtered = df.filter((df["Year"] >= 1965) & (df["Year"] <= 2016))

# Group by year and month, and count earthquakes
earthquake_count_by_month = df_filtered.groupBy("Year", "Month").agg(count("*").alias("EarthquakeCount"))

# Calculate the average frequency of earthquakes per month
total_months = (2016 - 1965 + 1) * 12  # Total number of months in the range
average_frequency = earthquake_count_by_month.selectExpr("avg(EarthquakeCount) as AverageFrequency").collect()[0]["AverageFrequency"]

print(f"Average frequency of earthquakes per month from 1965 to 2016: {average_frequency}")


# COMMAND ----------

df = df.dropna(subset=['Date']) # drop rows with null values in Date column
df = df.withColumn("Year", year("Date"))
earthquake_count_by_year = df.groupBy("Year").agg(count("*").alias("EarthquakeCount"))

# Sorting the dataframe in descending order of number of earthquakes for better understanding by using desc()
earthquake_count_by_year_ordered = earthquake_count_by_year.orderBy(desc("EarthquakeCount"))

# Displaying the dataframe
earthquake_count_by_year_ordered.show(n=30)
# Collect the results to the driver for visualization
results = earthquake_count_by_year.collect()

# Visualize the results (example using matplotlib)
years = [int(row["Year"]) for row in results]
counts = [row["EarthquakeCount"] for row in results]

# Create x coordinates for the bars.
x_pos = np.arange(len(years))
# plt.xticks(np.arange(1965, 2010, 10))

plt.bar(x_pos, counts, align='center')
plt.xticks(x_pos, years)

plt.xlabel("Year")
plt.ylabel("Number of Earthquakes")
plt.title("Number of Earthquakes by Year")
plt.show()




# COMMAND ----------

# Calculating the number of earthquakes in a year and sorting it in descending order

earthquakes_per_year = df.groupBy('Year').agg(count('*').alias('EarthquakeCount'))
earthquakes_per_year = earthquakes_per_year.orderBy('EarthquakeCount', ascending=False)
earthquakes_per_year.show()

# COMMAND ----------

import matplotlib.pyplot as pltt
average_magnitude_by_year = df.groupBy("Year").agg(mean("Magnitude").alias("AverageMagnitude"))
average_magnitude_by_year=average_magnitude_by_year.orderBy('AverageMagnitude','Year',ascending=False)
average_magnitude_by_year.display()

# COMMAND ----------

# MAGIC %md
# MAGIC From the above output we can conclude that the earthquakes of maximum magnitude on an average occurred in the year 1968 and with the minimum average magnitude in the year 1977. We can see the graphical demonstration of the same below.
# MAGIC There is no uniform pattern in the increase or decrease of magnitude as per the year.

# COMMAND ----------

average_magnitude_by_year = average_magnitude_by_year.toPandas()

import seaborn as sns

plt.figure(figsize=(12, 8))
sns.lineplot(x="Year", y="AverageMagnitude", data=average_magnitude_by_year)
plt.title('Average Magnitude of Earthquakes Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Magnitude')
plt.show()

# COMMAND ----------

# Understanding how year impact the standard deviation of the earthquakes

# Importing the necessary libraries like stddev here for calculating standard deviation
from pyspark.sql.functions import stddev

# Calculate standard deviation of earthquake magnitude by year
std_dev_by_year = df.groupBy("Year").agg(stddev("Magnitude").alias("MagnitudeStdDev"))

std_dev_by_year=std_dev_by_year.orderBy('MagnitudeStdDev',ascending=False)
# Show the result
std_dev_by_year.show(100)

# COMMAND ----------

import plotly.express as px
std_dev_by_year = df.groupBy("Year").agg(stddev("Magnitude").alias("MagnitudeStdDev"))

# Convert PySpark DataFrame to Pandas for visualization
pandas_std_dev_by_year = std_dev_by_year.toPandas()


plt.figure(figsize=(12, 8))
sns.lineplot(x="Year", y="MagnitudeStdDev", data=pandas_std_dev_by_year)
plt.title('Standard Deviation of Earthquake Magnitude Over the Years')
plt.xlabel('Year')
plt.ylabel('Magnitude Standard Deviation')
plt.show()



 






# COMMAND ----------

As we can see that the standard deviation does not remain relatively constant over the years, it may suggest inconsistent level of seismic activity or an unstable geological environment.Spikes or peaks in the standard deviation might indicate years with unusual or heightened seismic activity. This could be due to natural geological processes, tectonic plate movements, or other factors.Changes in standard deviation could also be influenced by human activities such as mining, reservoir-induced seismicity (changes in seismic activity due to the filling of large reservoirs), or other anthropogenic factors.




# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Geographical loactionca
# MAGIC Earthquakes are more likely to occur in certain regions due to the Earth's tectonic plate boundaries and geological characteristics. Here are some key points to consider:
# MAGIC
# MAGIC 1. Tectonic Plate Boundaries
# MAGIC 2. Ring of Fire
# MAGIC 3. Fault Lines
# MAGIC 4. Depth and Magnitude
# MAGIC 5. Human Activities
# MAGIC 6. Seismic Hazard Maps
# MAGIC
# MAGIC The longitude and latitude of a location can influence the depth and magnitude of earthquakes experienced in that region.
# MAGIC
# MAGIC In the below cell, I am calculating the number of earthquakes as per latitude and longitude.

# COMMAND ----------

from pyspark.sql.functions import count, col
import matplotlib.pyplot as plt
import seaborn as sns


# Group by location and count the number of earthquakes
earthquake_frequency_by_location = df.groupBy("Latitude", "Longitude").agg(count("*").alias("EarthquakeCount"))

# Show the result
earthquake_frequency_by_location.orderBy(col("EarthquakeCount").desc()).show()



# Convert PySpark DataFrame to Pandas for visualization
pandas_earthquake_frequency = earthquake_frequency_by_location.toPandas()

# Plotting
plt.figure(figsize=(12, 8))
sns.scatterplot(x="Longitude", y="Latitude", size="EarthquakeCount", data=pandas_earthquake_frequency)
plt.title('Earthquake Frequency by Location')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()




# COMMAND ----------

import plotly.express as px
import pandas as pd
 
fig = px.scatter_geo(df, lat='Latitude',
                     lon='Longitude',
                     color="Magnitude",
                     fitbounds='locations',
                     scope='asia')
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC From the above output we can conclude that the maximum number of earthquakes (which is 4) occured at  Latitude and Longitude of 51.5 and  -174.8 respectively.

# COMMAND ----------


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sqrt, mean

# Create a Spark session
spark = SparkSession.builder.appName("EarthquakeAnalysis").getOrCreate()


df = spark.read.csv('/mnt/mmmdata-dest/tmp/parul/database.csv', header=True, inferSchema=True)

# Select the 'Magnitude' column
magnitude_column = col('Magnitude')

# Calculate the root mean square for the 'Magnitude' column
rms_magnitude = df.select(sqrt(mean(magnitude_column ** 2)).alias('RMS_Magnitude')).collect()[0]['RMS_Magnitude']

print(f"Root Mean Square (RMS) of Magnitude: {rms_magnitude}")



# COMMAND ----------



# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pandas_earthquake_frequency['Longitude'] = pd.to_numeric(pandas_earthquake_frequency['Longitude'])
pandas_earthquake_frequency['Latitude'] = pd.to_numeric(pandas_earthquake_frequency['Latitude'])

plt.figure(figsize=(12, 8))
sns.jointplot(x="Longitude", y="Latitude", kind="hex", data=pandas_earthquake_frequency)
plt.title('Hexbin Plot of Earthquake Frequency by Geographic Location')
plt.show()

# COMMAND ----------


