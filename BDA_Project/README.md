# Retail Analytics with Hadoop & Pig

This project demonstrates a full end-to-end big data analytics workflow for retail transaction data using Hadoop and Pig. It covers ingestion, cleaning, analytics, and output storage, including daily/monthly sales, top products, store performance, discount analysis, and RFM customer segmentation.

---

## 1. HDFS Setup

Create directories in HDFS for raw, staged, and output data:

hdfs dfs -mkdir -p /data/retail/raw
hdfs dfs -mkdir -p /data/retail/staged/transactions_clean
hdfs dfs -mkdir -p /data/retail/outputs

Upload your CSV file:

hdfs dfs -put -f /local/path/transactions.csv /data/retail/raw/
hdfs dfs -ls /data/retail/raw

---

## 2. Pig Setup

Pig should be installed with access to Piggybank (piggybank.jar) for CSV parsing and date UDFs. You can run Pig in local mode for testing:

pig -x local script.pig

Or in MapReduce mode for cluster:

pig -x mapreduce script.pig -param INPUT=<HDFS_PATH> -param OUT=<HDFS_OUTPUT> -param PIGGY=<path_to_piggybank.jar>

---

## 3. Data Staging & Cleaning

stage_transactions.pig performs:

- CSV parsing and type casting  
- Trimming and validation  
- Date extraction (YYYY-MM-DD)  
- Computed fields: GrossAmount, CalcNetAmount  
- Validation flags  
- Output stored as tab-delimited text  

Run:

pig -x mapreduce stage_transactions.pig \
    -param INPUT=/data/retail/raw/transactions.csv \
    -param OUT=/data/retail/staged/transactions_clean \
    -param PIGGY=/path/to/piggybank.jar

---

## 4. Analytics Pig Jobs

### 4.1 Total Sales

- Daily and monthly totals  
- Stored in /data/retail/outputs/total_sales

pig -x mapreduce total_sales.pig \
    -param INPUT=/data/retail/staged/transactions_clean \
    -param OUT=/data/retail/outputs/total_sales

### 4.2 Top Products

- Revenue and quantity-based top products  
- Stored in /data/retail/outputs/by_product

pig -x mapreduce top_products.pig

### 4.3 Sales by Store & Category

- Revenue and quantity aggregated by store and product category  
- Stored in /data/retail/outputs/by_store_category

pig -x mapreduce store_category.pig

### 4.4 Discount Analysis

- Buckets discounts: 0–9%, 10–29%, 30–49%, 50+%  
- Computes average quantity, avg unit price, and total revenue per bucket  
- Stored in /data/retail/outputs/discount_analysis

pig -x mapreduce discount_analysis.pig

### 4.5 RFM Analysis

- Computes Recency (days since last purchase), Frequency, Monetary  
- Scores customers into 1–5 for R, F, M  
- Stored in /data/retail/outputs/rfm

pig -x mapreduce rfm.pig \
    -param INPUT=/data/retail/staged/transactions_clean \
    -param OUT=/data/retail/outputs/rfm \
    -param CURRENT_DATE=2025-09-20

---

## 5. Hive Integration (Optional but Recommended)

Create external Hive table for staged data:

CREATE EXTERNAL TABLE retail_clean (
  CustomerID BIGINT,
  ProductID STRING,
  Quantity INT,
  Price DOUBLE,
  GrossAmount DOUBLE,
  DiscountApplied_pct DOUBLE,
  CalcNetAmount DOUBLE,
  ReportedTotal DOUBLE,
  PaymentMethod STRING,
  StoreLocation STRING,
  ProductCategory STRING,
  txn_date STRING,
  ValidationFlag STRING
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
LOCATION '/data/retail/staged/transactions_clean';

Convert to Parquet for faster queries:

CREATE TABLE retail_parquet STORED AS PARQUET AS SELECT * FROM retail_clean;

---

## 6. Scheduling & Orchestration

Use Airflow or Oozie to automate:

1. Ingest CSV → HDFS  
2. Stage & clean data (stage_transactions.pig)  
3. Run analytics scripts  
4. Import results to Hive or export to BI tools  
5. Archive raw/staged data  

---

## 7. Validation & Testing

Test with a small sample CSV locally:

head -n 1000 transactions.csv > sample.csv
pig -x local stage_transactions.pig -param INPUT=sample.csv -param OUT=/tmp/sample_out -param PIGGY=/path/to/piggybank.jar

Verify outputs:

hdfs dfs -cat /data/retail/staged/transactions_clean/part-* | head -n 20

---

## 8. Performance Tips

- Compress outputs with Snappy or Deflate  
- Use columnar formats (Parquet/ORC)  
- Partition by txn_date for fast queries  
- Avoid small files (merge with hdfs dfs -getmerge)  
- Use combiners and tune reducers (set default_parallel in Pig)  
- Register UDFs for date parsing or custom scoring  

---

## 9. File Structure

/data/retail/  
  raw/  
  staged/  
  outputs/  
    total_sales/  
    by_product/  
    by_store_category/  
    discount_analysis/  
    rfm/  
  archive/  

Pig scripts:  
  stage_transactions.pig  
  total_sales.pig  
  top_products.pig  
  store_category.pig  
  discount_analysis.pig  
  rfm.pig  

---

## 10. References

- https://pig.apache.org/  
- https://pig.apache.org/docs/r0.17.0/udf.html  
- https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsUserGuide.html  
- https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL