# 🛍️ Retail Analytics with Hadoop & Pig

> A comprehensive big data analytics solution for retail transaction processing and insights generation

[![Hadoop](https://img.shields.io/badge/Hadoop-3.x-orange?logo=apache-hadoop)](https://hadoop.apache.org/)
[![Pig](https://img.shields.io/badge/Apache%20Pig-0.17-blue?logo=apache)](https://pig.apache.org/)
[![Hive](https://img.shields.io/badge/Apache%20Hive-3.x-yellow?logo=apache-hive)](https://hive.apache.org/)

## 📋 Overview

This project demonstrates a complete end-to-end big data analytics workflow for retail transaction data using the Hadoop ecosystem. The solution provides comprehensive insights into retail operations through automated data processing and advanced analytics.

### 🎯 Key Features

- **📊 Daily & Monthly Sales Analysis** - Track revenue trends over time
- **🏆 Top Products Identification** - Discover best-performing items
- **🏪 Store Performance Metrics** - Compare location effectiveness
- **💰 Discount Impact Analysis** - Measure promotional effectiveness
- **👥 RFM Customer Segmentation** - Understand customer behavior patterns

---

## 🚀 Quick Start

### Prerequisites

- Hadoop cluster (3.x+)
- Apache Pig (0.17+)
- Apache Hive (3.x+) - Optional
- piggybank.jar for advanced UDFs

---

## 🗂️ Project Structure

```
/data/retail/
├── raw/                    # Raw CSV files
├── staged/                 # Cleaned & processed data
│   └── transactions_clean/
├── outputs/               # Analytics results
│   ├── total_sales/
│   ├── by_product/
│   ├── by_store_category/
│   ├── discount_analysis/
│   └── rfm/
└── archive/               # Historical data

pig-scripts/
├── stage_transactions.pig
├── total_sales.pig
├── top_products.pig
├── store_category.pig
├── discount_analysis.pig
└── rfm.pig
```

---

## 🔧 Setup Instructions

### 1. 📁 HDFS Directory Setup

```bash
# Create required directories
hdfs dfs -mkdir -p /data/retail/{raw,staged/transactions_clean,outputs,archive}

# Upload transaction data
hdfs dfs -put -f /local/path/transactions.csv /data/retail/raw/

# Verify upload
hdfs dfs -ls /data/retail/raw
```

### 2. 🐷 Pig Environment Setup

```bash
# Local testing mode
pig -x local script.pig

# Production cluster mode
pig -x mapreduce script.pig \
    -param INPUT=/data/retail/raw/transactions.csv \
    -param OUT=/data/retail/staged/transactions_clean \
    -param PIGGY=/path/to/piggybank.jar
```

---

## 📈 Analytics Pipeline

### Phase 1: 🧹 Data Staging & Cleaning

The `stage_transactions.pig` script performs comprehensive data preparation:

```bash
pig -x mapreduce stage_transactions.pig \
    -param INPUT=/data/retail/raw/transactions.csv \
    -param OUT=/data/retail/staged/transactions_clean \
    -param PIGGY=/path/to/piggybank.jar
```

**Processing Steps:**
- ✅ CSV parsing with type casting
- ✂️ Data trimming and validation  
- 📅 Date standardization (YYYY-MM-DD)
- 🧮 Computed fields: GrossAmount, CalcNetAmount
- 🚩 Validation flags for data quality
- 💾 Output as tab-delimited format

### Phase 2: 📊 Core Analytics

#### 💹 Total Sales Analysis
```bash
pig -x mapreduce total_sales.pig \
    -param INPUT=/data/retail/staged/transactions_clean \
    -param OUT=/data/retail/outputs/total_sales
```

#### 🏆 Top Products Analysis  
```bash
pig -x mapreduce top_products.pig
```

#### 🏪 Store & Category Performance
```bash
pig -x mapreduce store_category.pig
```

#### 💰 Discount Impact Analysis
```bash
pig -x mapreduce discount_analysis.pig
```

#### 👥 RFM Customer Segmentation
```bash
pig -x mapreduce rfm.pig \
    -param INPUT=/data/retail/staged/transactions_clean \
    -param OUT=/data/retail/outputs/rfm \
    -param CURRENT_DATE=2025-09-20
```

---

## 🗃️ Hive Integration (Optional)

### Create External Table for Staged Data

```sql
CREATE EXTERNAL TABLE IF NOT EXISTS retail_clean (
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
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
LOCATION '/data/retail/staged/transactions_clean';
```

### Optimize with Parquet Format

```sql
CREATE TABLE IF NOT EXISTS retail_parquet
STORED AS PARQUET
AS SELECT * FROM retail_clean;
```

---

## 🔄 Automation & Orchestration

### Workflow Steps for Airflow/Oozie

1. **📥 Data Ingestion** → Upload CSV files to HDFS
2. **🧹 Data Staging** → Execute `stage_transactions.pig`  
3. **📊 Analytics Execution** → Run all analytics scripts
4. **🗃️ Data Integration** → Import results to Hive
5. **📦 Archival** → Move processed files to archive

---

## 🧪 Testing & Validation

### Local Testing with Sample Data

```bash
# Create sample dataset
head -n 1000 /local/path/transactions.csv > sample.csv

# Test locally
pig -x local stage_transactions.pig \
    -param INPUT=sample.csv \
    -param OUT=/tmp/sample_out \
    -param PIGGY=/path/to/piggybank.jar

# Verify results
hdfs dfs -cat /data/retail/staged/transactions_clean/part-* | head -n 20
```

---

## ⚡ Performance Optimization

### 🏎️ Best Practices

- **📦 Compression**: Use Snappy or Deflate for outputs
- **📊 Storage Format**: Implement Parquet/ORC for columnar efficiency
- **📁 Partitioning**: Partition by `txn_date` for query performance
- **🔗 File Management**: Merge small files using `hdfs dfs -getmerge`
- **⚙️ Parallelism**: Tune `default_parallel` setting in Pig
- **🔧 Custom UDFs**: Register specialized functions for date parsing

### 📊 Compression Example

```bash
# Enable compression in Pig
pig -Dpig.tmpfilecompression=true \
    -Dpig.tmpfilecompression.codec=gz \
    -x mapreduce script.pig
```

---

## 🔍 Monitoring & Troubleshooting

### Common Commands

```bash
# Check job progress
yarn application -list

# Monitor HDFS usage
hdfs dfs -du -h /data/retail/

# View Pig logs
tail -f /var/log/pig/pig.log
```

---

## 📚 Resources & Documentation

| Resource | Description | Link |
|----------|-------------|------|
| 🐷 **Apache Pig** | Official documentation | [pig.apache.org](https://pig.apache.org/) |
| 🧰 **Piggybank UDFs** | Built-in user-defined functions | [UDF Documentation](https://pig.apache.org/docs/r0.17.0/udf.html) |
| 🗃️ **Hadoop HDFS** | File system guide | [HDFS User Guide](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsUserGuide.html) |
| 🏛️ **Apache Hive** | External tables & Parquet | [Hive Language Manual](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL) |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙋‍♀️ Support

For questions and support:
- 📧 Email: [support@company.com](mailto:support@company.com)
- 📝 Issues: [GitHub Issues](https://github.com/yourorg/retail-analytics/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourorg/retail-analytics/discussions)

---

<div align="center">

**Built with ❤️ using the Hadoop Ecosystem**

[⭐ Star this repo](https://github.com/yourorg/retail-analytics) | [🍴 Fork it](https://github.com/yourorg/retail-analytics/fork) | [📝 Report Issues](https://github.com/yourorg/retail-analytics/issues)

</div>