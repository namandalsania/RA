import pandas as pd
from fuzzywuzzy import fuzz, process

# Load the dataset
file_path = 'C:/Users/naman/Documents/Data Science/RA/data/dataset_with_subcategories.csv'  
df = pd.read_csv(file_path)

# List of all categories to match
categories = [
    "inventory", "Asset", "Loans to others", "Payments to deposit", "Uncategorized Asset", 
    "Buildings", "Land", "office equipment", "Computers & tablets", "Copiers", 
    "Custom software or app", "Furniture", "Phones", "Photo & video equipment", 
    "Tools, machinery, and equipment", "Vehicles", "Customer prepayments", "sales tax payments", 
    "Short-term business loans-payments made", "Long-term business loans-payments made", 
    "Mortgages-payments made", "Federal estimated taxes paid", "Personal expenses:Federal taxes", 
    "Owner retirement plans", "Personal expenses", "State taxes", "Personal healthcare", 
    "Personal healthcare:Health insurance premiums", "Personal healthcare:HSA contributions", 
    "State estimated taxes", "Refunds to customers", "Cost of goods sold", "Equipment rental", 
    "Subcontractor expenses", "Supplies & materials", "Advertising & marketing", "Listing fees", 
    "Social media advertising & marketing", "Website ads", "Building & property rent", 
    "Business licences", "Commissions & fees", "Contract labor", "Contributions to charities", 
    "Employee benefits", "Employee retirement plans", "Employee benefits:Group term life insurance", 
    "Employee benefits:Health & accident plans", "Worker's compensation insurance", 
    "Entertainment with clients", "General business expenses", "Bad DebtGeneral", 
    "Bank fees & service charges", "Continuing education", "General business expenses:Memberships & subscriptions", 
    "Uniforms", "Liability insurance", "Property insurance", "Rental insurance", 
    "Business loan interest", "Credit card interest", "Mortgage interest", "Accounting fees", 
    "Legal fees", "Meals with clients", "Travel meals", "Merchant account fees", "Office supplies", 
    "Printing & photocopying", "Shipping & postage", "Small tools and equipment", 
    "Software & apps", "Wages", "Repairs & maintenance", "Supplies & materials", 
    "Payroll taxes", "Property taxes", "Airfare", "Hotels", "Taxis or shared rides", 
    "Vehicle rental", "Uncategorized Expense", "Disposal & waste feesUtilities", "ElectricityUtilities", 
    "Heating & coolingUtilities", "Internet & TV services", "Phone service", 
    "Utilities:Water & sewer", "Depreciation Home office", "Home office:Homeowner & rental insurance", 
    "Home office:Home utilities", "Home office:Mortgage interest", "Home office:Property taxes", 
    "Home office:Rent", "Home office:Repairs & maintenance", "Vehicle expenses", "Parking & tolls"
]

# Function to match category for "other"
def match_category(transaction_type, categories):
    # Use fuzzy matching to find the best match from the category list
    best_match, score = process.extractOne(transaction_type, categories, scorer=fuzz.token_sort_ratio)
    
    # Set a threshold to ensure only high-confidence matches are used
    if score > 60:  # This threshold can be adjusted
        return best_match
    else:
        return "other"

# Apply the matching logic to replace "other" in the 'category' column
df['category'] = df.apply(lambda row: match_category(row['transaction_type'], categories) 
                          if row['category'] == 'other' else row['category'], axis=1)

# Save the updated dataset
df.to_csv('C:/Users/naman/Documents/Data Science/RA/data/updated_dataset.csv', index=False)

# Print the updated dataset head
print(df.head())
