-- stage_transactions.pig
-- Usage:
-- pig -x mapreduce stage_transactions.pig -param INPUT=/data/retail/raw/transactions.csv -param OUT=/data/retail/staged/transactions_clean -param PIGGY=/path/to/piggybank.jar

%default INPUT '/data/retail/raw/Retail_Transaction_Dataset.csv'
%default OUT '/data/retail/staged/transactions_clean'
%default PIGGY '/home/steve/pig/contrib/piggybank/java/piggybank.jar'

REGISTER '$PIGGY';

-- Load CSV with quoted fields and multiline support
A = LOAD '$INPUT' USING org.apache.pig.piggybank.storage.CSVExcelStorage(',', 'YES_MULTILINE')
    AS (
        CustomerID:chararray, 
        ProductID:chararray, 
        Quantity:chararray, 
        Price:chararray, 
        TransactionDate:chararray, 
        PaymentMethod:chararray, 
        StoreLocation:chararray, 
        ProductCategory:chararray, 
        DiscountApplied_pct:chararray, 
        TotalAmount:chararray
    );

-- Cast types and trim whitespace
B = FOREACH A GENERATE
      (long)TRIM(CustomerID)           AS CustomerID,
      TRIM(ProductID)                  AS ProductID,
      (int)TRIM(Quantity)              AS Quantity,
      (double)TRIM(Price)              AS Price,
      TRIM(TransactionDate)            AS TransactionDate,
      TRIM(PaymentMethod)              AS PaymentMethod,
      TRIM(StoreLocation)              AS StoreLocation,
      TRIM(ProductCategory)            AS ProductCategory,
      ((double)TRIM(DiscountApplied_pct)) AS DiscountApplied_pct,
      (double)TRIM(TotalAmount)        AS TotalAmount;

-- Simple sanity filters
C = FILTER B BY CustomerID IS NOT NULL AND ProductID IS NOT NULL AND Quantity IS NOT NULL;

-- Extract date (YYYY-MM-DD) from TransactionDate
D = FOREACH C GENERATE
      CustomerID,
      ProductID,
      Quantity,
      Price,
      TotalAmount,
      DiscountApplied_pct,
      PaymentMethod,
      StoreLocation,
      ProductCategory,
      (SUBSTRING(TransactionDate,0,10)) AS txn_date;

-- Compute derived fields: GrossAmount and CalcNetAmount
E = FOREACH D GENERATE
      CustomerID,
      ProductID,
      Quantity,
      Price,
      (double)(Price * Quantity) AS GrossAmount,
      DiscountApplied_pct,
      (double)((Price * Quantity) * (1.0 - (DiscountApplied_pct/100.0))) AS CalcNetAmount,
      TotalAmount AS ReportedTotal,
      PaymentMethod,
      StoreLocation,
      ProductCategory,
      txn_date;

-- Validate: flag rows where calculated net and reported total differ
WITH_FLAGS = FOREACH E GENERATE
      *,
      (ABS(CalcNetAmount - ReportedTotal) <= 0.01 ? 'OK' : 'FLAG') AS ValidationFlag;

-- Store cleaned and flagged data as tab-delimited text
STORE WITH_FLAGS INTO '$OUT' USING PigStorage('\t', '-overwrite');