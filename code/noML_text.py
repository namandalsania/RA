import csv
import os

# Function to print the question and get the user's input
def ask_question(question):
    print(f"Computer: {question}")
    response = input("You: ").strip().lower()  # Using text input instead of speech
    return response
    
def save_to_csv(filename, fieldnames, data):
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(data)
    
# Define the complete list of categories and phrases
categories = {
    # Neither expense nor income
    'put_money_in_bank': [
        "put money in the bank", "deposit funds", "make a deposit", "made a deposit", 
        "add to account", "added to account", "save money", "saved money", "place in savings", 
        "placed in savings", "store cash", "credit account", "transfer to bank", 
        "bank a check", "banked a check", "fund account", "increase balance", 
        "increased balance", "bank deposit", "put in savings", "cash deposit", 
        "bank transfer"
    ],
    'took_money_out_of_bank': [
        "withdraw funds", "withdrew funds", "make a withdrawal", "made a withdrawal", 
        "cash out", "take out money", "took out money", "remove funds", "removed funds", 
        "debit account", "take money out", "took money out", "draw cash", 
        "withdrew cash", "withdraw cash", "bank withdrawal", "get cash", 
        "got cash", "take out cash", "took out cash", "reduce balance", 
        "reduced balance", "money withdrawal", "ATM withdrawal"
    ],
    'got_a_loan': [
        "secured a loan", "borrowed money", "took out a loan", "obtained financing", 
        "loan received", "credit received", "loan approval", "approved loan", 
        "financing obtained", "money borrowed", "loan granted", "got credit", 
        "loan disbursement", "loan acquisition", "loan initiation", "credit obtained"
    ],
    'received_a_line_of_credit': [
        "line of credit approval", "line of credit approved", "received a line of credit", 
        "credit line received", "credit extended", "got a credit line", 
        "credit line granted", "accessed credit line", "received credit extension", 
        "credit line acquisition", "opened a credit line", "line of credit initiated", 
        "credit line secured", "credit access", "granted line of credit", 
        "credit line approval", "line of credit obtained"
    ],
    'took_out_a_mortgage': [
        "secured a mortgage", "mortgage approval", "mortgage approved", 
        "got a mortgage", "mortgage obtained", "mortgage financing", 
        "mortgage loan received", "mortgage granted", "home loan received", 
        "house financing", "mortgage initiation", "mortgage acquired", 
        "home loan secured", "real estate financing", "mortgage loan granted", 
        "mortgage received"
    ],
    'paid_a_loan': [
        "loan repayment", "repaid a loan", "settled loan", "paid off debt", 
        "loan payoff", "cleared loan", "loan payment made", "paid back loan", 
        "loan discharge", "settled debt", "loan settlement", "paid off loan", 
        "loan clearance", "loan satisfied", "loan repayment completed"
    ],
    'bought_a_machine': [
        "purchased equipment", "acquired machinery", "bought equipment", 
        "machine acquisition", "equipment purchase", "obtained machinery", 
        "bought hardware", "equipment acquisition", "machine purchase", 
        "acquired hardware", "machinery purchase", "purchased machinery", 
        "bought device", "obtained equipment", "machine obtained"
    ],
    'bought_a_vehicle': [
        "purchased a car", "acquired a vehicle", "bought an automobile", 
        "vehicle purchase", "obtained a car", "acquired an automobile", 
        "bought a car", "vehicle acquisition", "car purchase", "bought a vehicle", 
        "obtained a vehicle", "automobile purchase", "acquired a car", 
        "vehicle obtained", "bought an automobile"
    ],
    'bought_real_estate': [
        "purchased property", "acquired real estate", "real estate purchase", 
        "bought property", "acquired property", "real estate acquisition", 
        "property purchase", "obtained real estate", "bought real property", 
        "real estate obtained", "property acquisition", "purchased real estate", 
        "acquired real property", "bought land", "property obtained"
    ],
    'bought_furniture': [
        "purchased furnishings", "purchased furniture", "acquired furniture", 
        "furniture purchase", "bought furnishings", "furniture acquisition", 
        "obtained furniture", "acquired furnishings", "bought household items", 
        "furniture obtained", "furnishings purchase", "acquired household items", 
        "purchased home items", "furniture procured", "home furnishings bought", 
        "bought home furnishings"
    ],
    'bought_a_copier': [
        "purchased a copier", "acquired a copier", "copier purchase", 
        "bought a copier", "copier acquisition", "obtained a copier", 
        "copier obtained", "acquired copy machine", "copier bought", 
        "purchased copy machine", "copier secured", "bought copy machine", 
        "copier acquired", "copy machine purchase", "obtained copy machine"
    ],
    'bought_a_computer': [
        "purchased a computer", "acquired a computer", "computer purchase", 
        "bought a computer", "computer acquisition", "obtained a computer", 
        "acquired a PC", "computer obtained", "bought a PC", "computer bought", 
        "PC purchase", "acquired laptop", "purchased a PC", "computer procured", 
        "PC obtained", "bought a MAC", "purchased a MAC"
    ],
    'bought_tools_or_equipment': [
        "purchased tools", "acquired equipment", "bought equipment", 
        "tools purchase", "equipment acquisition", "obtained tools", 
        "bought tools", "equipment purchase", "acquired tools", 
        "equipment obtained", "tools acquisition", "purchased equipment", 
        "equipment procured", "tools obtained", "tools bought"
    ],
    'paid_a_personal_expense': [
        "settled a personal expense", "paid a personal bill", "covered personal cost", 
        "personal expense paid", "cleared personal expense", "paid personal charges", 
        "personal bill paid", "expense settled", "paid personal outlay", 
        "personal cost covered", "personal expense settled", 
        "paid personal expenditure", "cleared personal bill", 
        "personal charges covered", "paid a personal fee"
    ],

    # Income
    'sold_asset': [
        "sold an asset", "disposed of an asset", "sold property", 
        "asset sale", "sold equipment", "divested asset", 
        "sold real estate", "asset disposal", "sold machinery", 
        "asset liquidation", "sold furniture", "sold possession", 
        "sold holdings", "asset divestment", "sold item", "sold belongings"
    ],
    'earned_interest': [
        "gained interest", "interest income", "received interest", 
        "interest earned", "interest accrued", "accrued interest", 
        "interest received", "interest generated", "interest obtained", 
        "interest yield", "interest collected", "interest profits", 
        "earned yield", "collected interest", "interest gain"
    ],
    'received_dividend': [
        "received a dividend", "dividend income", "got dividend", 
        "received payout", "dividend received", "dividend payment", 
        "dividend collected", "dividend gained", "collected dividend", 
        "earned dividend", "dividend obtained", "dividend profits", 
        "got payout", "dividend yield", "dividend accrued", 
        "received profits"
    ],
    'received_credit_card_rewards': [
        "credit card rewards", "rewards received", "got rewards", 
        "earned rewards", "rewards accrued", "collected rewards", 
        "credit rewards received", "credit card points", 
        "rewards gained", "received points", "credit points earned", 
        "rewards obtained", "points collected", "earned points", 
        "credit card benefits"
    ],
    'got_refund_of_expense': [
        "expense refund", "received refund", "refund received", 
        "refund obtained", "got reimbursement", "expense reimbursement", 
        "reimbursement received", "expense refunded", 
        "refund collected", "reimbursement gained", "refund given", 
        "got refund", "reimbursement obtained", "expense returned", 
        "received reimbursement"
    ],
    'got_other_income': [
        "other income", "additional income", "miscellaneous income", 
        "extra earnings", "other earnings", "supplementary income", 
        "additional earnings", "income received", "extra income", 
        "other revenue", "miscellaneous earnings", 
        "supplementary earnings", "additional revenue", "other profits", 
        "extra funds"
    ],
    'got_tax_refund': [
        "tax refund received", "tax rebate", "received tax refund", 
        "tax refund collected", "got tax return", "tax return received", 
        "rebate received", "tax repayment", "refund of taxes", 
        "received rebate", "tax refund obtained", "collected tax refund", 
        "tax refund given", "refund received", "tax return obtained"
    ],
    'got_partnership_income': [
        "partnership income", "LLC income", "income from partnership", 
        "earnings from LLC", "partnership earnings", "LLC earnings", 
        "received partnership income", "received LLC income", 
        "partnership profits", "LLC profits", "income from LLC", 
        "partnership revenue", "LLC revenue", "income from business", 
        "earnings from business"
    ],
    'sale_of_product_income': [
        "product sales revenue", "goods sales income", 
        "income from product sales", "revenue from goods sold", 
        "product revenue", "income from merchandise sales", 
        "goods income", "sales income from products", 
        "revenue from product transactions", 
        "income generated from product sales", 
        "product sales proceeds", "income from selling products", 
        "product sales profit", "revenue from selling goods", 
        "sales earnings from products"
    ],
    'sale_of_service_income': [
        "service sales revenue", "service income", "revenue from services", 
        "income from service sales", "service revenue", 
        "service sales income", "earnings from services", 
        "revenue generated from services", "service income stream", 
        "income from providing services", "service sales proceeds", 
        "service earnings", "revenue from service transactions", 
        "income generated from service sales", "service sales profit"
    ],

    # Expenses
    'inventory': [
        "stock", "goods", "supplies", "merchandise", "items", 
        "wares", "products", "commodities", "materials", 
        "inventory list", "inventory stock", "inventory items", 
        "inventory goods", "inventory supplies", "inventory materials"
    ],
    'asset': [
        "property", "resources", "holdings", "capital", "wealth", 
        "estate", "investment", "valuable", "possessions", 
        "goods", "funds", "assets", "asset base", "financial assets", 
        "owned items"
    ],
    'loans_to_others': [
        "credit to others", "money lent", "financial aid", 
        "funds lent", "loans given", "lending", "advances", 
        "borrowed funds", "lent money", "credit extended", 
        "loans granted", "loans made", "provided loans", 
        "loaned funds", "loaned money"
    ],
    'payments_to_deposit': [
        "deposits", "funds deposited", "cash deposits", 
        "money added", "payments made", "deposited funds", 
        "payments added", "bank deposits", "savings deposits", 
        "credit deposits", "payment deposits", "payment transfers", 
        "deposited payments", "account deposits", 
        "deposit transactions"
    ],
    'uncategorized_asset': [
        "miscellaneous asset", "unclassified asset", "general asset", 
        "various asset", "other asset", "unallocated asset", 
        "asset not classified", "undefined asset", "non-specific asset", 
        "unassigned asset", "asset not categorized", "assorted asset", 
        "diverse asset", "mixed asset", "uncategorized item"
    ],
    'buildings': [
        "structures", "facilities", "constructions", "properties", 
        "real estate", "edifices", "houses", "dwellings", 
        "establishments", "premises", "buildings", "complexes", 
        "architectures", "construction projects", "infrastructures"
    ],
    'land': [
        "property", "terrain", "ground", "real estate", "plots", 
        "fields", "farmland", "acreage", "lots", "territory", 
        "soil", "estate", "lands", "ground area", "land property"
    ],
    'office_equipment': [
        "office supplies", "office tools", "office gear", 
        "office machinery", "office devices", "workplace equipment", 
        "business equipment", "office apparatus", 
        "office furnishings", "office materials", "office instruments", 
        "office hardware", "office utilities", "office accessories", 
        "office implements"
    ],
    'computers_tablets': [
        "PCs", "laptops", "desktops", "notebooks", "computing devices", 
        "tablets", "digital devices", "electronic devices", 
        "portable computers", "tablet PCs", "computer systems", 
        "workstations", "computer hardware", "personal computers", 
        "handheld devices"
    ],
    'copiers': [
        "copy machines", "photocopiers", "duplicators", 
        "copying machines", "reproducers", "copy devices", 
        "xerox machines", "printers", "replicators", "copy equipment", 
        "copier machines", "copy hardware", "imaging devices", 
        "copy apparatus", "duplicating machines"
    ],
    'custom_software_app': [
        "bespoke software", "tailored software", "custom applications", 
        "personalized software", "customized apps", "custom programs", 
        "specialized software", "custom-built software", 
        "unique software", "custom-made software", 
        "custom software solutions", "specific software", 
        "software applications", "custom apps", "custom software tools"
    ],
    'furniture': [
        "furnishings", "household items", "home furniture", 
        "office furniture", "fixtures", "furniture pieces", 
        "interior furnishings", "furniture sets", "furniture items", 
        "furniture fittings", "furniture accessories", 
        "decorative furniture", "furniture products", 
        "furniture articles", "furniture goods"
    ],
    'phones': [
        "telephones", "cell phones", "mobile phones", "smartphones", 
        "handsets", "mobiles", "telephone devices", "telephone sets", 
        "communication devices", "cellular phones", 
        "portable phones", "wireless phones", "phones", 
        "mobile devices", "smart devices"
    ],
    'photo_video_equipment': [
        "cameras", "video cameras", "photography equipment", 
        "video gear", "photo gear", "imaging equipment", 
        "recording devices", "photo apparatus", "video apparatus", 
        "camera equipment", "photo tools", "video tools", 
        "camera gear", "photo devices", "video devices"
    ],
    'tools_machinery_equipment': [
        "tools", "machinery", "equipment", "machines", 
        "tools and machinery", "industrial tools", 
        "industrial machinery", "machinery and equipment", 
        "tools and equipment", "hardware", "machinery tools", 
        "mechanical equipment", "industrial equipment", 
        "work tools", "work machinery"
    ],
    'vehicles': [
        "automobiles", "cars", "trucks", "motor vehicles", 
        "vans", "SUVs", "buses", "motorcycles", "vehicles", 
        "transportation", "automobiles", "fleet", "sedans", 
        "coupes", "minivans"
    ],
    'customer_prepayments': [
        "advance payments", "prepaid customer", "prepayments", 
        "customer deposits", "customer advances", 
        "advance deposits", "early payments", "customer prepaids", 
        "prepaid advances", "upfront payments", "advanced payments", 
        "prepay", "prepayment receipts", "advance customer funds", 
        "customer prepayments"
    ],
    'sales_tax_payments': [
        "sales tax remittances", "tax payments", "tax remittances", 
        "sales tax dues", "sales tax disbursements", "tax settlements", 
        "tax transfers", "sales tax submissions", "tax contributions", 
        "sales tax liabilities", "sales tax fees", "tax outflows", 
        "tax expenditures", "sales tax obligations", "tax payables"
    ],
    'short_term_business_loans_payments_made': [
        "short-term loan repayments", "short-term debt payments", 
        "business loan payments", "short-term financing payments", 
        "short-term credit payments", "loan repayment", 
        "short-term borrowing payments", "loan installments", 
        "short-term loan outflows", "business debt payments", 
        "short-term loan servicing", "business loan settlements", 
        "loan paybacks", "short-term credit settlements", 
        "short-term loan clearances"
    ],
    'long_term_business_loans_payments_made': [
        "long-term loan repayments", "long-term debt payments", 
        "business loan settlements", "long-term financing payments", 
        "long-term credit payments", "loan installments", 
        "business debt servicing", "loan paybacks", 
        "long-term loan outflows", "business loan payments", 
        "long-term debt settlements", "loan servicing", 
        "long-term credit settlements", "business loan clearances", 
        "long-term loan servicing"
    ],
    'mortgages_payments_made': [
        "mortgage repayments", "mortgage payments", "home loan payments", 
        "mortgage settlements", "mortgage outflows", "loan repayments", 
        "mortgage dues", "mortgage paybacks", "mortgage clearances", 
        "house loan payments", "mortgage servicing", "mortgage payoffs", 
        "home loan repayments", "mortgage disbursements", 
        "mortgage liabilities"
    ],
    'federal_estimated_taxes_paid': [
        "federal tax payments", "estimated tax payments", 
        "federal tax dues", "tax outflows", "federal tax settlements", 
        "estimated tax remittances", "federal tax contributions", 
        "tax liabilities", "federal tax disbursements", "tax payments", 
        "federal tax transfers", "tax estimates paid", 
        "federal tax clearances", "tax outlays", "tax payables"
    ],
    'personal_expenses_federal_taxes': [
        "federal tax expenses", "personal tax payments", 
        "personal tax liabilities", "federal tax dues", "tax outflows", 
        "personal tax remittances", "federal tax contributions", 
        "personal tax expenditures", "federal tax settlements", 
        "tax obligations", "federal tax outlays", 
        "personal tax disbursements", "tax liabilities", "tax payments", 
        "federal tax outflows"
    ],
    'owner_retirement_plans': [
        "retirement plans", "owner pension plans", "retirement savings", 
        "owner 401(k)", "retirement accounts", "retirement benefits", 
        "owner retirement funds", "pension contributions", 
        "owner retirement savings", "retirement investments", 
        "pension plans", "retirement contributions", 
        "owner retirement accounts", "retirement funds", 
        "retirement provisions"
    ],
    'personal_expenses': [
        "personal expenditures", "personal costs", "personal outlays", 
        "personal payments", "individual expenses", "personal disbursements", 
        "personal liabilities", "personal spending", "personal bills", 
        "personal charges", "private expenses", "personal outflows", 
        "personal expenditures", "personal financial obligations", 
        "personal finance outlays"
    ],
    'state_taxes': [
        "state tax payments", "state tax dues", "state tax liabilities", 
        "state tax contributions", "state tax remittances", 
        "state tax outflows", "state tax settlements", 
        "state tax expenditures", "state tax obligations", 
        "state tax disbursements", "state tax payments", 
        "state tax outlays", "state tax clearances", 
        "state tax transfers", "state tax fees"
    ],
    'personal_healthcare': [
        "healthcare expenses", "medical costs", "healthcare payments", 
        "healthcare outlays", "medical expenditures", "healthcare spending", 
        "healthcare disbursements", "healthcare bills", "medical outflows", 
        "healthcare charges", "healthcare liabilities", "medical payments", 
        "healthcare expenses", "healthcare outflows", "medical outlays"
    ],
    'personal_healthcare_health_insurance_premiums': [
        "health insurance payments", "insurance premiums", 
        "medical insurance costs", "health insurance dues", "insurance outflows", 
        "medical insurance premiums", "health insurance contributions", 
        "insurance liabilities", "health insurance payments", 
        "medical insurance payments", "insurance expenditures", 
        "health insurance outlays", "medical insurance outflows", 
        "insurance premiums paid", "health insurance bills"
    ],
    'personal_healthcare_hsa_contributions': [
        "HSA payments", "health savings account contributions", 
        "HSA deposits", "HSA funding", "HSA contributions", 
        "health savings contributions", "HSA allocations", "HSA outflows", 
        "HSA payments", "health savings deposits", "HSA additions", 
        "health savings funding", "HSA investments", "HSA funds", 
        "HSA disbursements"
    ],
    'state_estimated_taxes': [
        "state tax estimates", "estimated state taxes", "state tax payments", 
        "estimated tax liabilities", "state tax outflows", "state tax dues", 
        "state tax remittances", "state tax contributions", 
        "estimated state tax payments", "state tax estimates paid", 
        "state tax disbursements", "estimated tax outflows", 
        "state tax outlays", "estimated tax remittances", 
        "state tax clearances"
    ],
    'refunds_to_customers': [
        "customer refunds", "customer reimbursements", "refunds issued", 
        "refunds given", "customer repayments", "customer returns", 
        "refund payments", "reimbursement payments", "refunds made", 
        "refund settlements", "customer credit", "refund disbursements", 
        "customer refund payments", "refund outflows", "refund transactions"
    ],
    'cost_of_goods_sold': [
        "COGS", "cost of sales", "goods costs", "production costs", 
        "inventory costs", "manufacturing expenses", "product costs", 
        "goods sold expenses", "sales costs", "goods expenditures", 
        "cost of products", "cost of inventory", "cost of merchandise", 
        "cost of production", "cost of goods"
    ],
    'equipment_rental': [
        "equipment lease", "equipment hire", "rented equipment", 
        "leased equipment", "equipment leasing", "rental equipment", 
        "hiring equipment", "equipment rental fees", "equipment leasing costs", 
        "equipment hire payments", "equipment leasing payments", 
        "equipment rental costs", "equipment rent", "equipment hiring fees", 
        "rented machinery"
    ],
    'subcontractor_expenses': [
        "subcontractor costs", "contractor payments", "outsourcing expenses", 
        "subcontractor fees", "contractor expenses", "subcontractor payments", 
        "outsourcing costs", "contractor fees", "contractor costs", 
        "subcontractor outlays", "outsourcing payments", "contractor disbursements", 
        "subcontractor liabilities", "subcontractor spending", 
        "contractor outflows"
    ],
    'supplies_and_materials': [
        "resources and materials", "equipment and supplies", 
        "inventory and materials", "materials and tools", 
        "components and supplies", "goods and materials", 
        "provisions and supplies", "inputs and materials", 
        "stocks and materials", "supplies and inventory", 
        "materials and resources", "supplies and components", 
        "tools and materials", "resources and inventory", 
        "equipment and materials"
    ],
    'advertising_and_marketing': [
        "promotion and marketing", "advertising and promotion", 
        "marketing and sales", "advertising and outreach", 
        "publicity and marketing", "marketing and advertising", 
        "promotion and advertising", "sales and marketing", 
        "advertising and branding", "marketing campaigns", 
        "advertising campaigns", "branding and promotion", 
        "advertising efforts", "marketing efforts", "promotion strategies"
    ],
    'listing_fees': [
        "registration fees", "enrollment fees", "subscription fees", 
        "entry fees", "sign-up fees", "admission fees", 
        "application fees", "catalog fees", "insertion fees", 
        "service fees", "posting fees", "commission fees", 
        "booking fees", "publication fees", "inclusion fees"
    ],
    'social_media_advertising_and_marketing': [
        "social media promotion", "social media campaigns", 
        "social media ads", "social media outreach", 
        "social media marketing", "social network advertising", 
        "social platform promotion", "social media branding", 
        "social media publicity", "social media sales", 
        "social media influence", "social media endorsements", 
        "social media exposure", "social media marketing campaigns", 
        "social media promotional efforts"
    ],
    'website_ads': [
        "web ads", "online ads", "digital ads", "internet ads", 
        "website banners", "webpage ads", "website advertising", 
        "online advertising", "digital advertising", "web marketing", 
        "internet advertising", "website promotions", 
        "online promotions", "digital promotions", "web promotions"
    ],
    'building_and_property_rent': [
        "real estate rent", "property lease", "building lease", 
        "rental expenses", "leasing costs", "office rent", 
        "facility rent", "space lease", "rental payments", 
        "renting expenses", "property rental", "leasing expenses", 
        "rent costs", "building rental", "property hiring"
    ],
    'business_licenses': [
        "commercial licenses", "enterprise licenses", "business permits", 
        "operating licenses", "trade licenses", "company licenses", 
        "business authorizations", "commercial permits", 
        "business certifications", "operation licenses", 
        "business registrations", "commercial authorizations", 
        "business approvals", "enterprise permits", "trade permits"
    ],
    'commissions_and_fees': [
        "service charges", "transaction fees", "processing fees", 
        "agent commissions", "broker fees", "commission payments", 
        "service fees", "handling fees", "commission charges", 
        "facilitation fees", "transaction charges", "intermediary fees", 
        "agent fees", "sales commissions", "broker commissions"
    ],
    'contract_labor': [
        "temporary labor", "contract workers", "freelance workers", 
        "contract staff", "temporary workers", "freelance labor", 
        "contracted personnel", "temporary staff", "freelancers", 
        "contract employees", "contractor labor", "hired hands", 
        "outsourced labor", "contract manpower", "temporary employees"
    ],
    'contributions_to_charities': [
        "charity donations", "charitable contributions", 
        "philanthropic donations", "donations to nonprofits", 
        "charity contributions", "philanthropic contributions", 
        "charity support", "donations to charities", 
        "charitable support", "philanthropy donations", 
        "charitable gifts", "charity funding", "nonprofit donations", 
        "charity aid", "philanthropic support"
    ],
    'employee_benefits': [
        "staff benefits", "worker benefits", "employee perks", 
        "employee incentives", "staff perks", "employee compensation", 
        "employee advantages", "staff incentives", "worker perks", 
        "employee rewards", "staff rewards", "employee welfare", 
        "worker advantages", "employee extras", "staff compensation"
    ],
    'employee_retirement_plans': [
        "pension plans", "retirement benefits", "employee pensions", 
        "staff retirement plans", "worker retirement plans", 
        "retirement schemes", "employee 401(k)", "staff pensions", 
        "retirement savings plans", "employee retirement schemes", 
        "staff retirement benefits", "worker pensions", 
        "retirement contributions", "employee retirement benefits", 
        "pension schemes"
    ],
    'employee_benefits_group_term_life_insurance': [
        "group life insurance", "term life insurance", 
        "employee life insurance", "staff life insurance", 
        "worker life insurance", "group term life coverage", 
        "term life coverage", "life insurance benefits", 
        "employee life coverage", "group life benefits", 
        "life insurance for staff", "worker life coverage", 
        "group term insurance", "life insurance plans", 
        "staff life coverage"
    ],
    'employee_benefits_health_and_accident_plans': [
        "health insurance", "accident insurance", "employee health plans", 
        "staff health insurance", "worker health insurance", 
        "health and accident coverage", "employee accident plans", 
        "staff accident insurance", "health benefits", 
        "accident benefits", "employee health coverage", 
        "worker health plans", "health and accident benefits", 
        "staff health plans", "employee accident coverage"
    ],
    'workers_compensation_insurance': [
        "employee compensation coverage", "workers' comp", 
        "work injury insurance", "employee injury insurance", 
        "workplace injury coverage", "compensation for workers", 
        "occupational injury insurance", "worker's accident insurance", 
        "work compensation policy", "workman's compensation insurance", 
        "employee compensation plan", "work injury protection", 
        "workplace accident coverage", "workers' injury coverage", 
        "workers' protection plan"
    ],
    'entertainment_with_clients': [
        "client entertainment", "customer hospitality", 
        "client engagement activities", "client leisure activities", 
        "client amusement", "customer engagement", 
        "client social events", "client activities", 
        "client leisure events", "customer amusement", 
        "client recreation", "customer outings", 
        "client hospitality events", "client interaction activities", 
        "client fun activities"
    ],
    'equipment_rental': [
        "equipment leasing", "rental of equipment", "tool rental", 
        "machine rental", "equipment hire", "renting machinery", 
        "leased equipment", "renting tools", "leasing machines", 
        "equipment hire service", "equipment lease", "machinery hire", 
        "tool hire", "equipment rental service", "machine hire"
    ],
    'general_business_expenses': [
        "business operating costs", "corporate expenses", 
        "company costs", "business expenditures", "corporate expenditures", 
        "business costs", "company expenses", "general business costs", 
        "general operating expenses", "business overheads", 
        "corporate overheads", "business spending", 
        "operational expenses", "corporate spending", 
        "company expenditures"
    ],
    'bad_debt_general': [
        "uncollectible accounts", "defaulted accounts", 
        "irrecoverable debt", "bad receivables", "nonrecoverable debt", 
        "unpaid debt", "bad loans", "defaulted loans", 
        "unrecoverable accounts", "bad credit", "delinquent debt", 
        "irrecoverable receivables", "noncollectable accounts", 
        "unpaid receivables", "unrecoverable loans"
    ],
    'bank_fees_and_service_charges': [
        "bank fees", "service charges", "banking fees", 
        "bank service costs", "account fees", "bank charges", 
        "banking service charges", "bank service fees", 
        "account service charges", "banking costs", 
        "account service fees", "financial service charges", 
        "banking service costs", "account charges", 
        "bank service expenses"
    ],
    'continuing_education': [
        "further education", "professional development", 
        "ongoing education", "continuous learning", "lifelong learning", 
        "adult education", "continuing studies", "professional training", 
        "advanced education", "career development", "skill enhancement", 
        "ongoing training", "extended education", "post-graduate education", 
        "vocational training"
    ],
    'memberships_and_subscriptions': [
        "membership fees", "subscription costs", "dues and subscriptions", 
        "association memberships", "magazine subscriptions", 
        "membership dues", "service subscriptions", 
        "organization memberships", "subscription fees", 
        "club memberships", "annual memberships", "monthly subscriptions", 
        "association dues", "subscription plans", "membership plans"
    ],
    'uniforms': [
        "work attire", "company uniforms", "work clothes", 
        "employee uniforms", "staff uniforms", "workwear", "work garments", 
        "company attire", "employee workwear", "staff attire", 
        "company work clothes", "work uniforms", "employee attire", 
        "staff workwear", "corporate uniforms"
    ],
    'liability_insurance': [
        "liability coverage", "liability protection", 
        "third-party insurance", "public liability insurance", 
        "business liability insurance", "general liability coverage", 
        "liability policy", "commercial liability insurance", 
        "liability indemnity", "liability risk insurance", 
        "liability assurance", "third-party coverage", 
        "business liability coverage", "general liability insurance", 
        "liability risk coverage"
    ],
    'property_insurance': [
        "property coverage", "property protection", 
        "real estate insurance", "building insurance", 
        "commercial property insurance", "property indemnity", 
        "asset insurance", "property risk insurance", 
        "real estate coverage", "property assurance", 
        "building coverage", "property risk coverage", 
        "commercial property coverage", "asset protection", 
        "real estate protection"
    ],
    'rental_insurance': [
        "renter's insurance", "tenant insurance", "lease insurance", 
        "rental coverage", "renter protection", "apartment insurance", 
        "rental property insurance", "tenant coverage", 
        "lease coverage", "rental protection plan", 
        "renter assurance", "tenant protection plan", "rental indemnity", 
        "rental risk insurance", "renter risk coverage"
    ],
    'business_loan_interest': [
        "business loan interest", "corporate loan interest", 
        "commercial loan interest", "business borrowing interest", 
        "company loan interest", "business financing interest", 
        "corporate borrowing interest", "commercial borrowing interest", 
        "business credit interest", "company borrowing interest", 
        "business debt interest", "corporate credit interest", 
        "commercial debt interest", "business loan interest charges", 
        "company loan interest charges"
    ],
    'credit_card_interest': [
        "credit card interest", "card interest", "credit interest", 
        "credit card finance charges", "card finance charges", 
        "credit card interest fees", "credit interest fees", "credit card APR", 
        "card interest rates", "credit card interest rates", 
        "credit card charges", "credit card finance fees", "credit card cost", 
        "credit card interest cost", "card interest cost"
    ],
    'mortgage_interest': [
        "mortgage interest", "home loan interest", "housing loan interest", 
        "mortgage finance charges", "mortgage interest rates", 
        "home loan interest rates", "mortgage interest fees", "mortgage APR", 
        "house loan interest", "mortgage interest payments", 
        "mortgage charges", "mortgage cost", "home loan charges", 
        "mortgage financing charges", "housing loan interest fees"
    ],
    'accounting_fees': [
        "bookkeeping fees", "accounting charges", "audit fees", 
        "financial statement fees", "CPA fees", "tax preparation fees", 
        "accounting service costs", "accountancy fees", "accountant fees", 
        "accounting expenses", "financial review fees", 
        "financial audit charges", "ledger maintenance fees", 
        "financial report fees", "accounting consultancy fees"
    ],
    'legal_fees': [
        "attorney fees", "lawyer charges", "legal service costs", 
        "litigation fees", "legal expenses", "court costs", 
        "legal consultancy fees", "legal charges", "legal advisory fees", 
        "legal representation costs", "legal filing fees", 
        "legal documentation fees", "legal counsel fees", 
        "legal aid costs", "legal retainers"
    ],
    'meals_with_clients': [
        "client lunches", "business meals", "client dinners", 
        "corporate dining", "client entertainment meals", 
        "business dining", "client breakfasts", "client brunches", 
        "client meal expenses", "meals for client meetings", 
        "dining with clients", "client hospitality meals", 
        "client meal costs", "client food expenses", "client dining expenses"
    ],
    'travel_meals': [
        "travel food expenses", "meals on business trips", "travel dining", 
        "meals during travel", "travel meal costs", 
        "food expenses while traveling", "dining on the go", "in-transit meals", 
        "meals for travelers", "travel snack expenses", "business travel meals", 
        "meals during business trips", "food costs on trips", 
        "travel eating expenses", "meals on the road"
    ],
    'merchant_account_fees': [
        "merchant processing fees", "merchant service charges", 
        "payment gateway fees", "credit card processing fees", 
        "merchant transaction fees", "merchant account costs", 
        "merchant bank charges", "online payment fees", 
        "merchant service fees", "merchant banking fees", 
        "merchant transaction costs", "merchant fee charges", 
        "merchant account expenses", "merchant account charges", 
        "merchant processing costs"
    ],
    'office_supplies': [
        "office materials", "stationery supplies", "office equipment", 
        "office necessities", "office tools", "office inventory", 
        "office provisions", "office stock", "office resources", 
        "workplace supplies", "office sundries", "office items", 
        "office goods", "office consumables", "office paraphernalia"
    ],
    'printing_and_photocopying': [
        "print services", "copying services", "document printing", 
        "photo copying", "print jobs", "document reproduction", 
        "copy services", "photocopy services", "printing costs", 
        "copying expenses", "print expenses", "document duplication", 
        "printing fees", "copy fees", "reproduction services"
    ],
    'shipping_and_postage': [
        "mailing costs", "delivery fees", "shipping charges", "postal fees", 
        "courier expenses", "freight costs", "postage expenses", 
        "shipment costs", "mailing expenses", "delivery expenses", 
        "shipping expenses", "postal charges", "courier fees", 
        "freight charges", "postage fees"
    ],
    'small_tools_and_equipment': [
        "hand tools", "small machinery", "minor equipment", 
        "tools and devices", "small implements", "portable equipment", 
        "small hardware", "utility tools", "equipment and tools", 
        "small apparatus", "basic tools", "small instruments", 
        "equipment tools", "handheld tools", "small gear"
    ],
    'software_and_apps': [
        "applications", "computer programs", "software tools", 
        "software solutions", "software applications", "mobile apps", 
        "desktop software", "software systems", "digital applications", 
        "software platforms", "programs and applications", 
        "software products", "technology solutions", "software suites", 
        "software packages"
    ],
    'wages': [
        "salaries", "pay", "earnings", "compensation", "remuneration", 
        "income", "employee pay", "worker wages", "paychecks", "salary", 
        "hourly pay", "wage payments", "staff wages", "wage earnings", 
        "wage compensation"
    ],
    'repairs_and_maintenance': [
        "upkeep", "maintenance services", "repair services", 
        "servicing", "maintenance work", "repair work", "maintenance costs", 
        "repair costs", "fixes", "routine maintenance", 
        "corrective maintenance", "repair expenses", "maintenance expenses", 
        "equipment repairs", "facility maintenance"
    ],
    'supplies_and_materials': [
        "materials and supplies", "consumables", "provisions", 
        "resources", "inventory", "goods and supplies", "materials", 
        "supply items", "supply stock", "raw materials", "supply inventory", 
        "materials stock", "supply resources", "necessary supplies", 
        "essential materials"
    ],
    'payroll_taxes': [
        "employee taxes", "wage taxes", "payroll levies", "payroll deductions", 
        "salary taxes", "employment taxes", "worker taxes", 
        "payroll contributions", "payroll charges", "staff taxes", 
        "payroll withholdings", "employee tax contributions", 
        "payroll tax payments", "payroll tax liabilities", 
        "wage tax deductions"
    ],
    'property_taxes': [
        "real estate taxes", "land taxes", "property levies", 
        "property assessments", "real property taxes", "property tax payments", 
        "estate taxes", "property tax bills", "property tax charges", 
        "realty taxes", "property tax obligations", "land tax payments", 
        "property tax contributions", "property tax expenses", 
        "property tax liabilities"
    ],
    'airfare': [
        "flight costs", "airline tickets", "flight tickets", 
        "air travel expenses", "plane tickets", "flight expenses", 
        "airline fares", "flight charges", "airline costs", 
        "air travel costs", "flight fares", "plane fares", 
        "airline expenses", "air travel fees", "plane travel costs"
    ],
    'hotels': [
        "lodging", "accommodations", "inns", "motels", "hotel stays", 
        "hotel accommodations", "places to stay", "hotel rooms", "resorts", 
        "guest houses", "bed and breakfasts", "suites", "lodgings", 
        "hotel facilities", "hostels"
    ],
    'taxis_or_shared_rides': [
        "taxi services", "ride sharing", "cab rides", "shared transportation", 
        "rideshare services", "taxicabs", "carpooling", "ride hailing", 
        "taxi rides", "uber rides", "lyft rides", "shared rides", 
        "cab services", "ride services", "taxi cabs"
    ],
    'vehicle_rental': [
        "car hire", "auto rental", "vehicle hire", "car leasing", 
        "renting a vehicle", "rental car", "car rental service", 
        "leasing a car", "hiring a car", "rent-a-car", "auto hire", 
        "leasing a vehicle", "hiring an automobile", "vehicle leasing", 
        "automobile rental"
    ],
    'uncategorized_expense': [
        "miscellaneous expense", "other costs", "unclassified expense", 
        "general expense", "unallocated expense", "various expenses", 
        "undefined costs", "unspecified expense", "unlabeled expense", 
        "catch-all expense", "misc expense", "random expense", 
        "miscellaneous costs", "uncategorized costs", "unassigned expense"
    ],
    'disposal_waste_fees_utilities': [
        "garbage fees", "trash disposal costs", "waste management fees", 
        "rubbish collection charges", "sanitation fees", 
        "refuse collection costs", "waste disposal charges", 
        "trash removal fees", "garbage collection costs", "refuse fees", 
        "waste disposal costs", "sanitation charges", "rubbish fees", 
        "trash fees", "waste removal charges"
    ],
    'electricity_utilities': [
        "power costs", "electric bill", "electricity charges", 
        "utility bill", "energy expenses", "electricity fees", 
        "power expenses", "electricity costs", "electricity payments", 
        "utility expenses", "energy costs", "electricity expenses", 
        "power bill", "energy bill", "utility charges"
    ],
    'heating_cooling_utilities': [
        "HVAC expenses", "temperature control costs", "heating expenses", 
        "cooling expenses", "climate control costs", 
        "air conditioning expenses", "heating and cooling costs", 
        "thermal management costs", "HVAC costs", 
        "temperature regulation costs", "heating bills", "cooling bills", 
        "HVAC charges", "climate control expenses", 
        "temperature control expenses"
    ],
    'internet_tv_services': [
        "internet charges", "TV subscription", "broadband costs", 
        "cable fees", "streaming services", "Wi-Fi expenses", 
        "television services", "ISP fees", "internet services", 
        "digital TV costs", "internet bills", "TV services", 
        "cable costs", "streaming fees", "internet and TV expenses"
    ],
    'phone_service': [
        "telephone charges", "mobile service", "phone bill", 
        "cellular service", "telecom expenses", "landline costs", 
        "mobile phone costs", "phone expenses", "cell phone fees", 
        "telephone service", "mobile charges", "cellular charges", 
        "phone costs", "telephone expenses", "telecom costs"
    ],
    'utilities_water_sewer': [
        "water bill", "sewer charges", "water utility", "sewer fees", 
        "water expenses", "sewage costs", "water and sewer costs", 
        "sewage expenses", "water utility charges", "sewer services", 
        "water services", "water and sewer expenses", "sewage bills", 
        "water fees", "sewer expenses"
    ],
    'depreciation_home_office': [
        "home office depreciation", "office at home depreciation", 
        "home workspace depreciation", "home office value reduction", 
        "office depreciation", "home office wear and tear", 
        "depreciation of home office", "workspace at home depreciation", 
        "home office asset depreciation", "home office amortization", 
        "depreciation on home office", "home office equipment depreciation", 
        "office use depreciation", "home office asset wear", 
        "home office devaluation"
    ],
    'home_office_homeowner_rental_insurance': [
        "home office insurance", "homeowner insurance for office", 
        "rental insurance for home office", "home office rental insurance", 
        "home office homeowner insurance", "office at home insurance", 
        "home workspace insurance", "home office coverage", 
        "home office property insurance", "home office liability insurance", 
        "home office protection", "insurance for home office", 
        "home office renters insurance", "home office homeowners insurance", 
        "home office rental coverage"
    ],
    'home_office_home_utilities': [
        "home office utilities", "utilities for home office", 
        "home office energy costs", "home office utility bills", 
        "office at home utilities", "home office electricity", 
        "home office water", "home office heating", "home office cooling", 
        "home office utility expenses", "home office power", 
        "home office gas", "utilities for office at home", 
        "home office water bill", "home office utility charges"
    ],
    'home_office_mortgage_interest': [
        "home office mortgage interest", "mortgage interest for home office", 
        "home office interest payments", "office at home mortgage interest", 
        "home office loan interest", "home office mortgage charges", 
        "interest on home office mortgage", 
        "home office mortgage interest expense", 
        "home workspace mortgage interest", 
        "home office interest costs", "interest payments for home office", 
        "home office mortgage interest fees", 
        "mortgage interest expense for home office", 
        "home office loan interest charges", "interest costs for home office"
    ],
    'home_office_property_taxes': [
        "home office property taxes", "property taxes for home office", 
        "home office tax payments", "office at home property taxes", 
        "home office real estate taxes", "home office property tax bills", 
        "property tax for home office", "home workspace property taxes", 
        "home office realty taxes", "home office property tax expenses", 
        "property tax payments for home office", 
        "home office property tax charges", "home office tax bills", 
        "property tax expenses for home office", "home office property tax fees"
    ],
    'home_office_rent': [
        "home office rent", "rent for home office", "office at home rent", 
        "home office rental", "home office lease", 
        "rent payments for home office", "home workspace rent", 
        "home office leasing", "rent expense for home office", 
        "home office rental payments", "rental cost for home office", 
        "home office rent expense", "home office lease payments", 
        "rent charges for home office", "home office rent fees"
    ],
    'home_office_repairs_maintenance': [
        "home office repairs", "home office maintenance", 
        "repairs for home office", "maintenance for home office", 
        "home office upkeep", "office at home repairs", 
        "home office fixings", "home workspace repairs", 
        "home office servicing", "home office repair costs", 
        "home office maintenance expenses", "home office mending", 
        "maintenance expenses for home office", "home office renovation", 
        "home office repair and upkeep"
    ],
    'vehicle_expenses': [
        "car expenses", "auto costs", "vehicle costs", "car maintenance", 
        "auto maintenance", "vehicle maintenance", "car upkeep", 
        "auto upkeep", "vehicle upkeep", "car running costs", "auto expenses", 
        "vehicle running costs", "car operation costs", "auto operation costs", 
        "vehicle operation costs"
    ],
    'parking_tolls': [
        "parking fees", "toll charges", "parking costs", "toll fees", 
        "parking expenses", "toll costs", "parking payments", "toll payments", 
        "parking charges", "toll expenses", "parking tickets", 
        "toll booths", "parking fares", "toll fares", "parking and toll costs"
    ]
}

# Function to classify the transaction text
def classify_text(text, categories):
    # Convert text to lowercase
    text = text.lower()

    # Iterate through each subcategory and its phrases
    for category, phrases in categories.items():
        for phrase in phrases:
            if phrase.lower() in text:
                return category  # Return the first matched category

    # Return 'uncategorized' if no match is found
    return 'uncategorized'

# Collect transaction details from user
def collect_transaction_data():
    transaction_details = {}
    
    response = ask_question("Did you buy something or pay someone?")
    if "yes" in response:
        path = 'expenses.csv'
        transaction_details['amount'] = ask_question("Okay, how much did you pay?")
        transaction_details['involvement'] = ask_question("Ok, whom did you pay?")
        transaction_details['payment_method'] = ask_question("Okay, what was the method of payment?")
        transaction_details['transaction_type'] = ask_question("Great, what was it for?")

    elif "no" in response:
        response = ask_question("Alright. Did you get paid for something?")
        if "yes" in response:
            type = 'income.csv'
            transaction_details['amount'] = ask_question("Okay, how much did you receive?")
            transaction_details['involvement'] = ask_question("Who paid you?")
            transaction_details['payment_method'] = ask_question("Great, how were you paid?")
            transaction_details['transaction_type'] = ask_question("Great. Was it a sale of a product or service or for something else?")

            if "product" in transaction_details['transaction_type']:
                path = 'income_product.csv'
            elif "service" in transaction_details['transaction_type']:
                path = 'income_service.csv'
            else:
                path = 'income_miscellaneous.csv'

        elif "no" in response:
            path = 'other_transactions.csv'
            transaction_details['amount'] = ask_question("Amounts involved?")
            transaction_details['involvement'] = ask_question("Who else was involved?")
            transaction_details['transaction_type'] = ask_question("Okay, then please describe the transaction.")
            
    else:
        print("Sorry, I did not understand your response.")
        
    return path, transaction_details or {}

# Main function to run the classification process
def main():
    # Step 1: Collect transaction details
    path, transaction_details = collect_transaction_data()
    transaction_type = transaction_details.get('transaction_type', '')

    # Step 2: Classify the transaction_type using the category list
    category = classify_text(transaction_type, categories)
    
    # Add the determined category to the transaction details
    transaction_details['category'] = category

    # Retain only the required fields
    filtered_transaction_details = {
        'amount': transaction_details['amount'],
        'involvement': transaction_details['involvement'],
        'payment_method': transaction_details['payment_method'],
        'transaction_type': transaction_details['transaction_type'],
        'category': transaction_details['category']
    }

    # Step 3: Save the result to a CSV file
    fieldnames = ['amount', 'involvement', 'payment_method', 'transaction_type', 'category']  # Required columns
    save_to_csv(path, fieldnames, filtered_transaction_details)

if __name__ == "__main__":
    main()
