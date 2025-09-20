-- total_sales.pig
-- Usage:
-- pig -x mapreduce total_sales.pig -param INPUT=/data/retail/staged/transactions_clean -param OUT=/data/retail/outputs/total_sales

%default INPUT '/data/retail/staged/transactions_clean'
%default OUT '/data/retail/outputs/total_sales'

-- Load staged transactions
data = LOAD '$INPUT' USING PigStorage('\t')
    AS (CustomerID:long, ProductID:chararray, Quantity:int, Price:double, GrossAmount:double, DiscountApplied_pct:double, CalcNetAmount:double, ReportedTotal:double, PaymentMethod:chararray, StoreLocation:chararray, ProductCategory:chararray, txn_date:chararray, ValidationFlag:chararray);

-- Filter only valid rows
valid_data = FILTER data BY ValidationFlag == 'OK';

-- Daily sales aggregation
grp_daily = GROUP valid_data BY txn_date;
daily_sales = FOREACH grp_daily GENERATE 
                 group AS txn_date, 
                 SUM(valid_data.CalcNetAmount) AS total_sales, 
                 COUNT(valid_data) AS txn_count;

STORE daily_sales INTO '$OUT/daily' USING PigStorage('\t');

-- Monthly sales aggregation (derive YYYY-MM)
monthly_prep = FOREACH valid_data GENERATE *, SUBSTRING(txn_date,0,7) AS txn_month;
grp_monthly = GROUP monthly_prep BY txn_month;
monthly_sales = FOREACH grp_monthly GENERATE 
                   group AS txn_month, 
                   SUM(monthly_prep.CalcNetAmount) AS total_sales, 
                   COUNT(monthly_prep) AS txn_count;

STORE monthly_sales INTO '$OUT/monthly' USING PigStorage('\t');