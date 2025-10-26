import pandas as pd
import re

# Load the CSV file
file_name = '/home/steve/Projects/Zoc/Ganesh Contacts - Sheet1.csv'
df = pd.read_csv(file_name)

# --- Use First Name Only ---
# This uses *only* the 'First Name' column.
df['Name'] = df['First Name'].fillna('').astype(str)

# Clean the name (remove special characters, extra spaces)
df['Name'] = df['Name'].str.replace(r'[©👑]', '', regex=True).str.strip()

# --- Consolidate Phone Numbers ---
# Aggregate all phone columns into one temporary column
phone_cols = ['Phone 1 - Value', 'Phone 2 - Value', 'Phone 3 - Value', 'Phone 4 - Value']
df['Numbers_Raw'] = df[phone_cols].fillna('').astype(str).agg(' ::: '.join, axis=1)

# --- Find, Separate, and Standardize Numbers ---

# Find all sequences of 8 or more digits
df['Number_List'] = df['Numbers_Raw'].str.findall(r'(\d{8,})')

# Create a new row for each number found (if a contact has 2 numbers, it becomes 2 rows)
df = df.explode('Number_List')

# Drop any rows that didn't have a number
df = df.dropna(subset=['Number_List'])

# --- FIX: Reset index ---
# This prevents the "duplicate labels" error after exploding.
df = df.reset_index(drop=True) 

# Rename for clarity
df = df.rename(columns={'Number_List': 'Number'})
df['Number'] = df['Number'].astype(str)

# Create a new column for the final, standardized number
df['Clean_Number'] = df['Number']

# --- Standardization Logic ---

# Rule 1: 10 digits (assume Indian)
# Find rows where 'Number' has length 10
is_10 = df['Number'].str.len() == 10
# For those rows, set 'Clean_Number' to '+91' + the number
df.loc[is_10, 'Clean_Number'] = '+91' + df['Number']

# Rule 2: 12 digits starts with 91 (assume Indian)
is_12_91 = (df['Number'].str.len() == 12) & df['Number'].str.startswith('91')
df.loc[is_12_91, 'Clean_Number'] = '+' + df['Number']

# Rule 3: 11-15 digits (assume international w/ code already included)
# This excludes numbers that already matched Rule 1 or 2
is_intl = df['Number'].str.len().between(11, 15) & ~is_10 & ~is_12_91
df.loc[is_intl, 'Clean_Number'] = '+' + df['Number']
# --- End Logic ---

# --- Finalize & Deduplicate ---

# Select only the 3 columns you requested
final_df = df[['Name', 'Course', 'Clean_Number']]

# Rename 'Clean_Number' to 'Number' for the final file
final_df = final_df.rename(columns={'Clean_Number': 'Number'})

# Clean up 'Course' column (fill blanks, remove spaces)
final_df['Course'] = final_df['Course'].fillna('').astype(str).str.strip()

# Remove rows with blank names (these are usually junk rows)
final_df = final_df[final_df['Name'].str.strip() != '']

# Drop any fully duplicate rows (same name, course, and number)
final_df = final_df.drop_duplicates()

# --- Save to new CSV ---
output_file = 'whatsapp_contacts_firstname_cleaned.csv'
final_df.to_csv(output_file, index=False)

# Print a confirmation and a sample
print(f"Cleaned data (First Name only) saved to {output_file}")
print("\n--- Sample of Cleaned Data ---")
print(final_df.head())
print(f"\nTotal contacts processed: {len(final_df)}")