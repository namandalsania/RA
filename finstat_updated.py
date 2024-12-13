# Code updated for the new expenses only dataset. Specifically altered for Prof. Swenson's PC

import pandas as pd
import re
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from datetime import datetime
import os

# Function to extract amounts from text using a regular expression
def extract_amount(statement):
    pattern = r'[\$€£₹]?(\d+\.?\d*)\s?(dollars?|euros?|rupees?|pounds?|bucks?)?'
    match = re.search(pattern, statement.lower())
    if match:
        return float(match.group(1))
    return None

# Define categories for mapping transactions
income_categories = [
    'sale of product income', 'sale of service income', 'credit card rewards',
    'insurance claims', 'interest earned', 'sale of an asset', 'other income',
    'earned interest', 'received dividend'
]

expense_categories = [
    'advertising & marketing', 'office supplies', 'wages', 'rent',
    'social media advertising & marketing', 'website ads', 'listing fees',
    'building & property rent', 'business licenses', 'commissions & fees',
    'contract labor', 'contributions to charities', 'employee benefits',
    'group term life insurance', 'health & accident plans', 'worker’s compensation insurance',
    'entertainment with clients', 'general business expenses', 'bank fees & service charges',
    'continuing education', 'uniforms', 'liability insurance', 'property insurance',
    'rental insurance', 'business loan interest', 'credit card interest', 'mortgage interest',
    'accounting fees', 'legal fees', 'meals with clients', 'travel meals', 'merchant account fees',
    'small tools and equipment', 'software & apps', 'payroll taxes', 'airfare', 'hotels',
    'taxis or shared rides', 'vehicle rental', 'utilities', 'disposal & waste fees', 'electricity',
    'heating & cooling', 'internet & tv services', 'phone service', 'water & sewer',
    'home office expenses', 'homeowner & rental insurance', 'home utilities', 'mortgage interest',
    'property taxes', 'home office rent', 'repairs & maintenance', 'personal expenses',
    'federal taxes', 'state taxes', 'health insurance premiums', 'HSA contributions',
    'refunds to customers', 'cost of goods sold', 'equipment rental', 'subcontractor expenses',
    'supplies & materials'
]

asset_categories = [
    'inventory', 'asset', 'loans to others', 'payments to deposit', 'uncategorized asset',
    'buildings', 'land', 'office equipment', 'computers & tablets', 'copiers', 'custom software or app',
    'furniture', 'phones', 'photo & video equipment', 'tools machinery equipment', 'vehicles',
    'customer prepayments'
]

liability_categories = [
    'sales tax payments', 'short_term_business_loans_payments_made',
    'long_term_business_loans_payments_made', 'mortgages_payments_made',
    'federal estimated taxes paid', 'personal expenses:federal taxes',
    'state taxes', 'state estimated taxes'
]

notes_categories = [
    'received a line of credit', 'took out a loan', 'took out a mortgage',
    'put money in bank', 'took money out of bank'
]

# Load the dataset
df = pd.read_csv('latest.csv')

# Prepare dictionaries for Income Statement, Balance Sheet, and Cash Flow
income_statement = {
    'Revenue': [],
    'Expenses': [],
    'Total Revenue': 0,
    'Total Expenses': 0,
    'Net Income': 0
}

balance_sheet = {
    'Assets': [],
    'Liabilities': [],
    'Total Assets': 0,
    'Total Liabilities': 0,
    'Equity': 0
}

cash_flow_statement = {
    'Cash Inflows': [],
    'Cash Outflows': [],
    'Total Inflows': 0,
    'Total Outflows': 0,
    'Net Cash Flow': 0
}

notes_to_financials = []

income_count = 0
expense_count = 0
asset_count = 0
liability_count = 0
notes_count = 0
uncategorized_count = 0

# Process each row in the dataset
for index, row in df.iterrows():
    transaction_type = row['Category'].lower()
    raw_amount = row['Amount']
    
    amount = extract_amount(raw_amount)

    if amount is None:
        continue
    
    # Prepare row details for Cash Flow statement with exact column names
    transaction_details = {
        "date": row["Date"],  # Ensuring 'date' key is added
        "type": transaction_type,
        "amount": amount,
        "full_details": {
            "Date": row["Date"],
            "Amount": raw_amount,
            "Involvement": row["Involvement"],
            "Payment Method": row["Payment Method"],
            "Transaction Type": row["Transaction Type"],
            "Category": row["Category"]
        }
    }

    # Income Transactions
    if transaction_type in income_categories:
        income_statement['Revenue'].append(transaction_details)
        balance_sheet['Assets'].append(transaction_details)
        cash_flow_statement['Cash Inflows'].append(transaction_details["full_details"])
        
        income_statement['Total Revenue'] += amount 
        balance_sheet['Total Assets'] += amount
        cash_flow_statement['Total Inflows'] += amount
        
    # Expense Transactions
    elif transaction_type in expense_categories:
        income_statement['Expenses'].append(transaction_details)
        balance_sheet['Assets'].append({'date': row['Date'], 'type': 'cash', 'amount': -amount})
        cash_flow_statement['Cash Outflows'].append(transaction_details["full_details"])
        
        income_statement['Total Expenses'] += amount
        balance_sheet['Total Assets'] -= amount
        cash_flow_statement['Total Outflows'] += amount
    
    # Asset Transactions
    elif transaction_type in asset_categories:
        balance_sheet['Assets'].append(transaction_details)
        balance_sheet['Total Assets'] += amount

    # Liability Transactions
    elif transaction_type in liability_categories:
        balance_sheet['Liabilities'].append({'date': row['Date'], 'type': transaction_type, 'amount': -amount})
        balance_sheet['Total Liabilities'] -= amount

    # Transactions that go into Notes
    elif transaction_type in notes_categories:
        notes_to_financials.append(transaction_details["full_details"])
        
    else:
        # Uncategorized transactions go to Notes
        notes_to_financials.append(transaction_details["full_details"])

# Function to group by type and sum the amounts
def group_and_sum(data):
    grouped_data = {}
    for item in data:
        transaction_type = item.get('type') or item.get('Transaction Type')
        amount = item.get('amount') if 'amount' in item else extract_amount(item.get('Amount', ''))
        
        if amount is None:
            continue
        
        if transaction_type in grouped_data:
            grouped_data[transaction_type] += amount
        else:
            grouped_data[transaction_type] = amount
    
    return [{'type': k, 'amount': v} for k, v in grouped_data.items()]

# Group and sum transactions
income_statement['Revenue'] = group_and_sum(income_statement['Revenue'])
income_statement['Expenses'] = group_and_sum(income_statement['Expenses'])
balance_sheet['Assets'] = group_and_sum(balance_sheet['Assets'])
balance_sheet['Liabilities'] = group_and_sum(balance_sheet['Liabilities'])

# Calculate Net Income, Owner's Equity, and Net Cash Flow
income_statement['Net Income'] = income_statement['Total Revenue'] - income_statement['Total Expenses']
balance_sheet['Equity'] = balance_sheet['Total Assets'] - balance_sheet['Total Liabilities']
cash_flow_statement['Net Cash Flow'] = cash_flow_statement['Total Inflows'] - cash_flow_statement['Total Outflows']

# Write Income Statement to Excel
income_statement_totals = [
    {'type': 'Total Revenue', 'amount': income_statement['Total Revenue']},
    {'type': 'Total Expenses', 'amount': income_statement['Total Expenses']},
    {'type': 'Net Income', 'amount': income_statement['Net Income']}
]

# Write Balance Sheet to Excel
balance_sheet_totals = [
    {'type': 'Total Assets', 'amount': balance_sheet['Total Assets']},
    {'type': 'Total Liabilities', 'amount': balance_sheet['Total Liabilities']},
    {'type': 'Equity', 'amount': balance_sheet['Equity']}
]

def write_income_statement_to_excel(filename, data):
    wb = Workbook()
    ws = wb.active
    ws.title = "Income Statement"

    # Title
    ws.append(["Income Statement"])
    ws.merge_cells("A1:B1")
    ws["A1"].font = Font(bold=True, size=14)
    ws["A1"].alignment = Alignment(horizontal="center")

    # Headers for Revenue Section
    ws.append(["Transaction Type", "Amount"])

    # Revenue Section
    for item in data["Revenue"]:
        ws.append([item.get("type", ""), item.get("amount", 0)])
        
    ws.append(["Total Revenue", data["Total Revenue"]])
    last_row = ws.max_row
    ws[f"A{last_row}"].font = Font(bold=True)
    ws[f"B{last_row}"].font = Font(bold=True)

    # Blank row for spacing
    ws.append([])

    # Headers for Expense Section
    ws.append(["Transaction Type", "Amount"])
    
    # Expenses Section
    for item in data["Expenses"]:
        ws.append([item.get("type", ""), item.get("amount", 0)])
        
    ws.append(["Total Expenses", data["Total Expenses"]])
    last_row = ws.max_row
    ws[f"A{last_row}"].font = Font(bold=True)
    ws[f"B{last_row}"].font = Font(bold=True)

    # Blank row for spacing
    ws.append([])

    # Net Income Section
    ws.append(["Net Income", data["Net Income"]])
    last_row = ws.max_row
    ws[f"A{last_row}"].font = Font(bold=True)
    ws[f"B{last_row}"].font = Font(bold=True)

    # Save the workbook
    wb.save(filename)

def write_balance_sheet_to_excel(filename, data):
    wb = Workbook()
    ws = wb.active
    ws.title = "Balance Sheet"

    # Title
    ws.append(["Balance Sheet"])
    ws.merge_cells("A1:B1")
    ws["A1"].font = Font(bold=True, size=14)
    ws["A1"].alignment = Alignment(horizontal="center")

    # Assets Section
    ws.append(["Transaction Type", "Amount"])
    
    for item in data["Assets"]:
        ws.append([item.get("type", ""), item.get("amount", 0)])
        
    ws.append(["Total Assets", data["Total Assets"]])
    last_row = ws.max_row
    ws[f"A{last_row}"].font = Font(bold=True)
    ws[f"B{last_row}"].font = Font(bold=True)
    
    # Blank row for spacing
    ws.append([])

    # Liabilities Section
    ws.append(["Transaction Type", "Amount"])
    
    for item in data["Liabilities"]:
        ws.append([item.get("type", ""), item.get("amount", 0)])
        
    ws.append(["Total Liabilities", data["Total Liabilities"]])
    last_row = ws.max_row
    ws[f"A{last_row}"].font = Font(bold=True)
    ws[f"B{last_row}"].font = Font(bold=True)
    
    # Blank row for spacing
    ws.append([])

    # Equity Section
    ws.append(["Equity", data["Equity"]])
    last_row = ws.max_row
    ws[f"A{last_row}"].font = Font(bold=True)
    ws[f"B{last_row}"].font = Font(bold=True)

    wb.save(filename)

# Updated formatting function for Cash Flow Statement
def write_cash_flow_to_excel(filename, data):
    wb = Workbook()
    ws = wb.active
    ws.title = "Cash Flow Statement"

    # Title
    ws.append(["Cash Flow Statement"])
    ws.merge_cells("A1:F1")
    ws["A1"].font = Font(bold=True, size=14)
    ws["A1"].alignment = Alignment(horizontal="center")
    
    # Helper function to parse dates for sorting
    def parse_date(item):
        try:
            return datetime.strptime(item.get("Date", ""), "%m-%d-%Y")
        except ValueError:
            return datetime.min  # Fallback for invalid/missing dates
        
    # Sort Cash Inflows and Cash Outflows by date in descending order
    sorted_inflows = sorted(data["Cash Inflows"], key=parse_date, reverse=True)
    sorted_outflows = sorted(data["Cash Outflows"], key=parse_date, reverse=True)

    # Cash Inflows
    ws.append(["Cash Inflows"])
    ws["A2"].font = Font(bold=True)
    ws.append(["Date", "Amount", "Involvement", "Payment Method", "Transaction Type", "Category"])
    
    for item in sorted_inflows:
        ws.append([
            item.get("Date", ""), 
            item.get("Amount", ""), 
            item.get("Involvement", ""), 
            item.get("Payment Method", ""), 
            item.get("Transaction Type", ""), 
            item.get("Category", "")
        ])
        
    ws.append(["Total Cash Inflows", data["Total Inflows"]])
    last_row = ws.max_row
    ws[f"A{last_row}"].font = Font(bold=True)
    ws[f"B{last_row}"].font = Font(bold=True)

    # Cash Outflows
    ws.append([])
    ws.append(["Cash Outflows"])
    ws[f"A{ws.max_row}"].font = Font(bold=True)  
    ws.append(["Date", "Amount", "Involvement", "Payment Method", "Transaction Type", "Category"])
    
    for item in sorted_outflows:
        ws.append([
            item.get("Date", ""), 
            item.get("Amount", ""), 
            item.get("Involvement", ""), 
            item.get("Payment Method", ""), 
            item.get("Transaction Type", ""), 
            item.get("Category", "")
        ])
        
    ws.append(["Total Cash Outflows", data["Total Outflows"]])
    last_row = ws.max_row
    ws[f"A{last_row}"].font = Font(bold=True)
    ws[f"B{last_row}"].font = Font(bold=True)

    # Net Cash Flow
    ws.append([])
    ws.append(["Net Cash Flow", data["Net Cash Flow"]])
    last_row = ws.max_row
    ws[f"A{last_row}"].font = Font(bold=True)
    ws[f"B{last_row}"].font = Font(bold=True)

    wb.save(filename)
    
# Create a directory named 'Financial Statements' if it doesn't exist
output_dir = "Financial Statements"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Save the updated Excel files
write_income_statement_to_excel(os.path.join(output_dir, "Income_Statement.xlsx"), income_statement)
write_balance_sheet_to_excel(os.path.join(output_dir, "Balance_Sheet.xlsx"), balance_sheet)
write_cash_flow_to_excel(os.path.join(output_dir, "Cash_Flow_Statement.xlsx"), cash_flow_statement)

# Saving notes to financials with full details in Excel
def write_notes_to_financials(filename, notes):
    wb = Workbook()
    ws = wb.active
    ws.title = "Notes to Financials"
    
    # Helper function to parse dates for sorting
    def parse_date(note):
        try:
            return datetime.strptime(note.get("Date", ""), "%m-%d-%Y")
        except ValueError:
            return datetime.min  # Fallback for invalid/missing dates

    # Sort notes by Date in descending order
    sorted_notes = sorted(notes, key=parse_date, reverse=True)

    headers = ["Date", "Amount", "Involvement", "Payment Method", "Transaction Type", "Category"]
    ws.append(headers)
    
    # Write sorted notes to the sheet
    for note in sorted_notes:
        ws.append([
            note.get("Date", ""),
            note.get("Amount", ""),
            note.get("Involvement", ""),
            note.get("Payment Method", ""),
            note.get("Transaction Type", ""),
            note.get("Category", "")
        ])

    wb.save(filename)
    
# Write Notes to Financials to Excel
write_notes_to_financials(os.path.join(output_dir, "Notes_to_Financials.xlsx"), notes_to_financials)

print("Income Statement, Balance Sheet, Cash Flow Statement, and Notes to Financials generated successfully.")