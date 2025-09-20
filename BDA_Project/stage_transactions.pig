%default INPUT '/home/steve/data/Retail_Transaction_Dataset.csv'
%default OUT '/home/steve/data/staged_transactions'

-- Remove existing output directory
sh rm -r -f '$OUT';

-- Load CSV
raw_data = LOAD '$INPUT' USING PigStorage(',')
    AS (CustomerID:int, ProductID:chararray, Quantity:int, Price:double,
        TransactionDate:chararray, PaymentMethod:chararray, StoreLocation:chararray,
        ProductCategory:chararray, DiscountApplied_pct:double, TotalAmount:double);

-- Optional: filter invalid rows
clean_data = FILTER raw_data BY TotalAmount >= 0;

-- Store cleaned/staged data
STORE clean_data INTO '$OUT' USING PigStorage('\t');