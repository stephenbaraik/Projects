%default INPUT '/home/steve/data/staged_transactions'
%default OUT '/home/steve/data/outputs/total_sales'

-- Remove existing output directories
sh rm -r -f '$OUT/daily';
sh rm -r -f '$OUT/monthly';

-- Load staged data
data = LOAD '$INPUT' USING PigStorage('\t')
    AS (CustomerID:int, ProductID:chararray, Quantity:int, Price:double,
        TransactionDate:chararray, PaymentMethod:chararray, StoreLocation:chararray,
        ProductCategory:chararray, DiscountApplied_pct:double, TotalAmount:double);

-- Daily sales aggregation
grp_daily = GROUP data BY TransactionDate;
daily_sales = FOREACH grp_daily GENERATE
                 group AS txn_date,
                 SUM(data.TotalAmount) AS total_sales,
                 COUNT(data) AS txn_count;

STORE daily_sales INTO '$OUT/daily' USING PigStorage('\t');

-- Monthly sales aggregation (YYYY-MM)
monthly_prep = FOREACH data GENERATE *, SUBSTRING(TransactionDate,0,7) AS txn_month;
grp_monthly = GROUP monthly_prep BY txn_month;
monthly_sales = FOREACH grp_monthly GENERATE
                   group AS txn_month,
                   SUM(monthly_prep.TotalAmount) AS total_sales,
                   COUNT(monthly_prep) AS txn_count;

STORE monthly_sales INTO '$OUT/monthly' USING PigStorage('\t');