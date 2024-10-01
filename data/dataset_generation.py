# # import pandas as pd
# # import random

# # # Define categories
# # categories_general = [
# #     "put money in bank", "took money out of bank", "got a loan",
# #     "received a line of credit", "took out a mortgage", "paid a loan",
# #     "bought a machine", "bought a vehicle", "bought real estate",
# #     "bought furniture", "bought a copier", "bought a computer",
# #     "bought tools or equipment", "paid a personal expense"
# # ]

# # categories_something_else = [
# #     "sold an asset", "earned interest", "received dividend",
# #     "received credit card rewards", "got refund of expense",
# #     "got some other type of income", "got a tax refund",
# #     "got income from a partnership or LLC"
# # ]

# # categories_purchase = [
# #     "inventory", "Asset", "Loans to others", "Payments to deposit",
# #     "Uncategorized Asset", "Buildings", "Land", "office equipment",
# #     "Computers & tablets", "Copiers", "Custom software or app",
# #     "Furniture", "Phones", "Photo & video equipment", "Tools, machinery, and equipment",
# #     "Vehicles", "Customer prepayments", "sales tax payments",
# #     "Short-term business loans-payments made", "Long-term business loans-payments made",
# #     "Mortgages-payments made", "Federal estimated taxes paid",
# #     "Personal expenses:Federal taxes", "Owner retirement plans",
# #     "Personal expenses", "State taxes", "Personal healthcare",
# #     "Personal healthcare:Health insurance premiums", "Personal healthcare:HSA contributions",
# #     "State estimated taxes", "Refunds to customers", "Cost of goods sold",
# #     "Equipment rental", "Subcontractor expenses", "Supplies & materials",
# #     "Advertising & marketing", "Listing fees", "Social media advertising & marketing",
# #     "Website ads", "Building & property rent", "Business licences",
# #     "Commissions & fees", "Contract labor", "Contributions to charities",
# #     "Employee benefits", "Employee retirement plans", "Employee benefits:Group term life insurance",
# #     "Employee benefits:Health & accident plans", "Worker's compensation insurance",
# #     "Entertainment with clients", "Equipment rental", "General business expenses",
# #     "Bad DebtGeneral", "Bank fees & service charges", "Continuing education",
# #     "General business expenses:Memberships & subscriptions", "Uniforms",
# #     "Liability insurance", "Property insurance", "Rental insurance",
# #     "Business loan interest", "Credit card interest", "Mortgage interest",
# #     "Accounting fees", "Legal fees", "Meals with clients", "Travel meals",
# #     "Merchant account fees", "Office supplies", "Printing & photocopying",
# #     "Shipping & postage", "Small tools and equipment", "Software & apps",
# #     "Wages", "Repairs & maintenance", "Supplies & materials",
# #     "Payroll taxes", "Property taxes", "Airfare", "Hotels", "Taxis or shared rides",
# #     "Vehicle rental", "Uncategorized Expense", "Disposal & waste feesUtilities",
# #     "ElectricityUtilities", "Heating & coolingUtilities", "Internet & TV services",
# #     "Phone service", "Utilities:Water & sewer", "Depreciation Home office",
# #     "Home office:Homeowner & rental insurance", "Home office:Home utilities",
# #     "Home office:Mortgage interest", "Home office:Property taxes",
# #     "Home office:Rent", "Home office:Repairs & maintenance", "Vehicle expenses",
# #     "Parking & tolls"
# # ]

# # payment_types = [
# #     "cash", "credit card", "Visa", "master card", "American express",
# #     "discover", "Venmo", "Paypal", "check", "cashiers check",
# #     "square", "merchant one", "clover", "apple pay", "ACH",
# #     "debit card", "google pay", "cash app", "echeck", "stripe"
# # ]

# # # Define sample inputs
# # amounts = [
# #     "$100", "150 dollars", "2000 USD", "$5000", "300 dollars",
# #     "$750", "1200 USD", "$4000", "10000 dollars", "250 USD"
# # ]

# # involvements = [
# #     "ABC Bank", "XYZ Corp", "John Doe", "Jane Smith", "ACME Corp",
# #     "Global Industries", "Big Bank", "Retail Store", "Real Estate Inc", "Tech Solutions"
# # ]

# # # Generate synthetic data
# # data = {
# #     "Amount Description": [],
# #     "Involvement Description": [],
# #     "Payment Description": [],
# #     "Category": []
# # }

# # num_samples = 100000  # Adjust as needed

# # for _ in range(num_samples):
# #     scenario = random.choice(["general", "product_payment", "service_payment", "something_else", "purchase"])
    
# #     if scenario == "general":
# #         category = random.choice(categories_general)
# #         amount = random.choice(amounts)
# #         involvement = random.choice(involvements)
# #         payment = random.choice(payment_types)
# #     elif scenario == "product_payment":
# #         category = "Sale of product income"
# #         amount = random.choice(amounts)
# #         involvement = random.choice(involvements)
# #         payment = random.choice(payment_types)
# #     elif scenario == "service_payment":
# #         category = "Sale of service income"
# #         amount = random.choice(amounts)
# #         involvement = random.choice(involvements)
# #         payment = random.choice(payment_types)
# #     elif scenario == "something_else":
# #         category = random.choice(categories_something_else)
# #         amount = random.choice(amounts)
# #         involvement = random.choice(involvements)
# #         payment = random.choice(payment_types)
# #     elif scenario == "purchase":
# #         category = random.choice(categories_purchase)
# #         amount = random.choice(amounts)
# #         involvement = random.choice(involvements)
# #         payment = random.choice(payment_types)
    
# #     data["Amount Description"].append(f"I paid {amount}")
# #     data["Involvement Description"].append(f"Involved with {involvement}")
# #     data["Payment Description"].append(f"Paid via {payment}")
# #     data["Category"].append(category)

# # df = pd.DataFrame(data)

# # # Save the dataset to a CSV file
# # dataset_file_path = 'synthetic_transactions_dataset_large.csv'
# # df.to_csv(dataset_file_path, index=False)

# # print(f"Dataset saved to {dataset_file_path}")

# import pandas as pd
# import random

# # Define categories
# categories_general = [
#     "put money in bank", "took money out of bank", "got a loan",
#     "received a line of credit", "took out a mortgage", "paid a loan",
#     "bought a machine", "bought a vehicle", "bought real estate",
#     "bought furniture", "bought a copier", "bought a computer",
#     "bought tools or equipment", "paid a personal expense"
# ]

# categories_product_income = ["Sale of Product Income"]
# categories_service_income = ["Sale of Service Income"]
# categories_something_else = [
#     "sold an asset", "earned interest", "received dividend",
#     "received credit card rewards", "got refund of expense",
#     "got some other type of income", "got a tax refund",
#     "got income from a partnership or LLC"
# ]

# categories_purchase = [
#     "inventory", "Asset", "Loans to others", "Payments to deposit",
#     "Uncategorized Asset", "Buildings", "Land", "office equipment",
#     "Computers & tablets", "Copiers", "Custom software or app",
#     "Furniture", "Phones", "Photo & video equipment", "Tools, machinery, and equipment",
#     "Vehicles", "Customer prepayments", "sales tax payments",
#     "Short-term business loans-payments made", "Long-term business loans-payments made",
#     "Mortgages-payments made", "Federal estimated taxes paid",
#     "Personal expenses:Federal taxes", "Owner retirement plans",
#     "Personal expenses", "State taxes", "Personal healthcare",
#     "Personal healthcare:Health insurance premiums", "Personal healthcare:HSA contributions",
#     "State estimated taxes", "Refunds to customers", "Cost of goods sold",
#     "Equipment rental", "Subcontractor expenses", "Supplies & materials",
#     "Advertising & marketing", "Listing fees", "Social media advertising & marketing",
#     "Website ads", "Building & property rent", "Business licences",
#     "Commissions & fees", "Contract labor", "Contributions to charities",
#     "Employee benefits", "Employee retirement plans", "Employee benefits:Group term life insurance",
#     "Employee benefits:Health & accident plans", "Worker's compensation insurance",
#     "Entertainment with clients", "Equipment rental", "General business expenses",
#     "Bad DebtGeneral", "Bank fees & service charges", "Continuing education",
#     "General business expenses:Memberships & subscriptions", "Uniforms",
#     "Liability insurance", "Property insurance", "Rental insurance",
#     "Business loan interest", "Credit card interest", "Mortgage interest",
#     "Accounting fees", "Legal fees", "Meals with clients", "Travel meals",
#     "Merchant account fees", "Office supplies", "Printing & photocopying",
#     "Shipping & postage", "Small tools and equipment", "Software & apps",
#     "Wages", "Repairs & maintenance", "Supplies & materials",
#     "Payroll taxes", "Property taxes", "Airfare", "Hotels", "Taxis or shared rides",
#     "Vehicle rental", "Uncategorized Expense", "Disposal & waste feesUtilities",
#     "ElectricityUtilities", "Heating & coolingUtilities", "Internet & TV services",
#     "Phone service", "Utilities:Water & sewer", "Depreciation Home office",
#     "Home office:Homeowner & rental insurance", "Home office:Home utilities",
#     "Home office:Mortgage interest", "Home office:Property taxes",
#     "Home office:Rent", "Home office:Repairs & maintenance", "Vehicle expenses",
#     "Parking & tolls"
# ]

# payment_types = [
#     "cash", "credit card", "Visa", "master card", "American express",
#     "discover", "Venmo", "Paypal", "check", "cashiers check",
#     "square", "merchant one", "clover", "apple pay", "ACH",
#     "debit card", "google pay", "cash app", "echeck", "stripe"
# ]

# # Define sample inputs
# amounts = [
#     "$100", "150 dollars", "2000 USD", "$5000", "300 dollars",
#     "$750", "1200 USD", "$4000", "10000 dollars", "250 USD"
# ]

# involvements = [
#     "ABC Bank", "XYZ Corp", "John Doe", "Jane Smith", "ACME Corp",
#     "Global Industries", "Big Bank", "Retail Store", "Real Estate Inc", "Tech Solutions"
# ]

# # Define templates for variable responses
# response_templates = {
#     "general": [
#         "I transferred {amount} to {involvement}.",
#         "Today, I deposited {amount} at {involvement}.",
#         "I paid {amount} to {involvement} for services.",
#         "I sent {amount} to {involvement}.",
#         "I paid {amount} to {involvement} via {payment}."
#     ],
#     "product_income": [
#         "I received {amount} from {involvement} for selling a product.",
#         "{involvement} paid me {amount} for a product.",
#         "Sale of product: {involvement} gave me {amount}.",
#         "I got {amount} from {involvement} for the sale of a product."
#     ],
#     "service_income": [
#         "I received {amount} from {involvement} for services rendered.",
#         "{involvement} paid me {amount} for services.",
#         "Service income: {involvement} gave me {amount}.",
#         "I got {amount} from {involvement} for the sale of services."
#     ],
#     "something_else": [
#         "I received {amount} from {involvement} for {category}.",
#         "{involvement} paid me {amount} for {category}.",
#         "{category}: {involvement} gave me {amount}.",
#         "I got {amount} from {involvement} for {category}."
#     ],
#     "purchase": [
#         "I bought something for {amount} from {involvement}.",
#         "{involvement} sold me an item for {amount}.",
#         "I paid {amount} to {involvement} for a purchase.",
#         "I bought {category} from {involvement} for {amount}."
#     ]
# }

# # Generate synthetic data
# data = {
#     "Amount Description": [],
#     "Involvement Description": [],
#     "Payment Description": [],
#     "Transaction Type": []
# }

# num_samples = 100000  # Adjust as needed

# for _ in range(num_samples):
#     scenario = random.choice(["general", "product_income", "service_income", "something_else", "purchase"])
    
#     if scenario == "general":
#         category = random.choice(categories_general)
#         amount = random.choice(amounts)
#         involvement = random.choice(involvements)
#         payment = random.choice(payment_types)
#         response = random.choice(response_templates[scenario]).format(amount=amount, involvement=involvement, payment=payment)
#     elif scenario == "product_income":
#         category = "Sale of Product Income"
#         amount = random.choice(amounts)
#         involvement = random.choice(involvements)
#         payment = random.choice(payment_types)
#         response = random.choice(response_templates[scenario]).format(amount=amount, involvement=involvement, payment=payment)
#     elif scenario == "service_income":
#         category = "Sale of Service Income"
#         amount = random.choice(amounts)
#         involvement = random.choice(involvements)
#         payment = random.choice(payment_types)
#         response = random.choice(response_templates[scenario]).format(amount=amount, involvement=involvement, payment=payment)
#     elif scenario == "something_else":
#         category = random.choice(categories_something_else)
#         amount = random.choice(amounts)
#         involvement = random.choice(involvements)
#         payment = random.choice(payment_types)
#         response = random.choice(response_templates[scenario]).format(amount=amount, involvement=involvement, category=category)
#     elif scenario == "purchase":
#         category = random.choice(categories_purchase)
#         amount = random.choice(amounts)
#         involvement = random.choice(involvements)
#         payment = random.choice(payment_types)
#         response = random.choice(response_templates[scenario]).format(amount=amount, involvement=involvement, category=category)
    
#     data["Amount Description"].append(response)
#     data["Involvement Description"].append(involvement)
#     data["Payment Description"].append(payment)
#     data["Transaction Type"].append(category)

# df = pd.DataFrame(data)

# # Save the dataset to a CSV file
# dataset_file_path = 'dataset.csv'
# df.to_csv(dataset_file_path, index=False)

# print(f"Dataset saved to {dataset_file_path}")

import csv
import random

# Predefined categories with example sentences
payment_methods = [
    "I paid using cash", "I used my credit card", "I paid with Visa", "I used my master card", 
    "I used my American express", "I paid with discover", "I transferred money using Venmo", 
    "I paid through Paypal", "I paid by check", "I used a cashiers check", "I used square for payment", 
    "I used merchant one for the transaction", "I paid using clover", "I used apple pay", 
    "I used ACH for the payment", "I paid with my debit card", "I used google pay", 
    "I paid using cash app", "I used an echeck", "I used stripe for payment"
]

transaction_types_payment = [
    "This was for inventory", "It was an Asset", "Loans to others", "Payments to deposit", 
    "It was for an Uncategorized Asset", "I paid for Buildings", "The payment was for Land", 
    "It was for office equipment", "I bought Computers & tablets", "I paid for Copiers", 
    "The payment was for Custom software or app", "I bought Furniture", "I paid for Phones", 
    "I bought Photo & video equipment", "The payment was for Tools, machinery, and equipment", 
    "I bought Vehicles", "I made Customer prepayments", "The payment was for sales tax payments", 
    "I paid for Short-term business loans-payments made", "It was for Long-term business loans-payments made", 
    "I paid for Mortgages-payments made", "The payment was for Federal estimated taxes paid", 
    "It was for Personal expenses:Federal taxes", "I paid for Owner retirement plans", "It was a Personal expense", 
    "I paid for State taxes", "The payment was for Personal healthcare", 
    "I paid for Personal healthcare:Health insurance premiums", "It was for Personal healthcare:HSA contributions", 
    "I paid State estimated taxes", "The payment was for Refunds to customers", 
    "The payment was for Cost of goods sold", "It was for Equipment rental", 
    "I paid for Subcontractor expenses", "The payment was for Supplies & materials", 
    "It was for Advertising & marketing", "I paid for Lissting fees", 
    "The payment was for Social media advertising & marketing", "It was for Website ads", 
    "I paid for Building & property rent", "The payment was for Business licences", 
    "I paid for Commissions & fees", "It was for Contract labor", "The payment was for Contributions to charities", 
    "It was for Employee benefits", "I paid for Employee retirement plans", 
    "The payment was for Employee benefits:Group term life insurance", 
    "It was for Employee benefits:Health & accident plans", "I paid for Worker's compensation insurance", 
    "The payment was for Entertainment with clients", "It was for Equipment rental", 
    "I paid for General business expenses", "The payment was for Bad DebtGeneral", 
    "I paid for Bank fees & service charges", "The payment was for Continuing education", 
    "It was for General business expenses:Memberships & subscriptions", "I paid for Uniforms", 
    "The payment was for Liability insurance", "It was for Property insurance", 
    "I paid for Rental insurance", "The payment was for Business loan interest", 
    "I paid Credit card interest", "The payment was for Mortgage interest", "I paid for Accounting fees", 
    "The payment was for Legal fees", "It was for Meals with clients", "I paid for Travel meals", 
    "The payment was for Merchant account fees", "It was for Office supplies", 
    "I paid for Printing & photocopying", "The payment was for Shipping & postage", 
    "It was for Small tools and equipment", "I paid for Software & apps", "The payment was for Wages", 
    "It was for Repairs & maintenance", "I paid for Supplies & materials", 
    "The payment was for Payroll taxes", "It was for Property taxes", "I paid for Airfare", 
    "The payment was for Hotels", "It was for Taxis or shared rides", "I paid for Vehicle rental", 
    "The payment was for Uncategorized Expense", "I paid for Disposal & waste feesUtilities", 
    "The payment was for ElectricityUtilities", "It was for Heating & coolingUtilities", 
    "I paid for Internet & TV services", "The payment was for Phone service", 
    "I paid for Utilities:Water & sewer", "The payment was for Depreciation Home office", 
    "It was for Home office:Homeowner & rental insurance", "I paid for Home office:Home utilities", 
    "The payment was for Home office:Mortgage interest", "I paid for Home office:Property taxes", 
    "It was for Home office:Rent", "I paid for Home office:Repairs & maintenance", 
    "The payment was for Vehicle expenses", "I paid for Parking & tolls"
]

transaction_types_income = [
    "I sold an asset (machine, furniture, real estate, etc)", "I earned interest", "I received dividend", 
    "I received credit card rewards", "I got refund of expense", "I got some other type of income", 
    "I got a tax refund", "I got income from a partnership or LLC"
]

transaction_types_other = [
    "I put money in bank", "I took money out of bank", "I got a loan", "I received a line of credit", 
    "I took out a mortgage", "I paid a loan", "I bought a machine", "I bought a vehicle", 
    "I bought real estate", "I bought furniture", "I bought a copier", "I bought a computer", 
    "I bought tools or equipment", "I paid a personal expense"
]

# Sample user responses for full sentences
amount_descriptions = [
    "I paid {} to {}", "I received {} from {}", "The amount involved was {} with {}", 
    "A transaction of {} was made to {}", "An amount of {} was transferred to {}"
]

involvements = ["my friend", "my colleague", "a company", "a store", "a vendor"]

# Function to generate a random transaction detail with full sentences for the amount
def generate_transaction_detail():
    response = random.choice(["yes", "no"])
    
    if response == "yes":  # expense or income
        response = random.choice(["yes", "no"])
        
        if response == "yes":  # buy or pay
            amount = random.choice(amount_descriptions).format(random.choice(["100 dollars", "500 dollars", "1500 dollars", "2000 dollars", "2500 dollars"]), random.choice(involvements))
            involvement = f"I paid the amount to {random.choice(involvements)}"
            payment_method = random.choice(payment_methods)
            transaction_type = f"{random.choice(transaction_types_payment)}"
            
            return {
                'amount': amount,
                'involvement': involvement,
                'payment_method': payment_method,
                'transaction_type': transaction_type
            }
            
        else:
            transaction_type = random.choice(transaction_types_income)
            amount = random.choice(amount_descriptions).format(random.choice(["100 dollars", "500 dollars", "1500 dollars", "2000 dollars", "2500 dollars"]), random.choice(involvements))
            involvement = f"I received the amount from {random.choice(involvements)}"
            payment_method = random.choice(payment_methods)
            
            if "product" in transaction_type:
                transaction_type = "It was for the sale of a product"
            elif "service" in transaction_type:
                transaction_type = "It was for a service I rendered"
                
            return {
                'amount': amount,
                'involvement': involvement,
                'payment_method': payment_method,
                'transaction_type': transaction_type
            }
            
    else:
        amount = random.choice(amount_descriptions).format(random.choice(["100 dollars", "500 dollars", "1500 dollars", "2000 dollars", "2500 dollars"]), random.choice(involvements))
        involvement = f"The amount involved {random.choice(involvements)}"
        payment_method = random.choice(payment_methods)
        transaction_type = f"{random.choice(transaction_types_other)}"
        
        return {
            'amount': amount,
            'involvement': involvement,
            'payment_method': payment_method,
            'transaction_type': transaction_type
        }

# Generating the dataset
def generate_dataset(num_rows, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['amount', 'involvement', 'payment_method', 'transaction_type'])
        
        for _ in range(num_rows):
            transaction_detail = generate_transaction_detail()
            
            writer.writerow([
                transaction_detail['amount'],
                transaction_detail['involvement'],
                transaction_detail['payment_method'],
                transaction_detail['transaction_type']
            ])

# Generate 100,000 rows dataset
generate_dataset(100000, 'dataset.csv')


