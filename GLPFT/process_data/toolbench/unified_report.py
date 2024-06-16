import csv

def parse_api_categories(file_path):
    api_categories = {}
    with open(file_path, 'r') as f:
        for line in f:
            api_fam, category = line.strip().split(': ')
            api_categories[api_fam] = category
    return api_categories

def parse_api_report(file_path):
    certain_apis = {}
    all_apis = {}
    section = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            if line == '===== CERTAIN APIS =====':
                section = 'certain'
                next(f)  # Skip the header
            elif line == '===== ALL APIS =====':
                section = 'all'
                next(f)  # Skip the header
            elif section == 'certain' or section == 'all':
                if ', ' in line:
                    api_full_name, count = line.split(', ')
                    count = int(count)
                    if '_for_' in api_full_name:
                        endpoint, api_fam = api_full_name.rsplit('_for_', 1)
                        if section == 'certain':
                            certain_apis[api_fam] = certain_apis.get(api_fam, {})
                            certain_apis[api_fam][endpoint] = count
                        elif section == 'all':
                            all_apis[api_fam] = all_apis.get(api_fam, {})
                            all_apis[api_fam][endpoint] = count

    return certain_apis, all_apis

def generate_csv(output_file, api_categories, certain_apis, all_apis):
    header = ['endpoint', 'api_family', 'category', 'number_of_samples', 'number_of_certain_samples']
    rows = []

    for api_fam, endpoints in all_apis.items():
        category = api_categories.get(api_fam, 'Unknown')
        for endpoint, count in endpoints.items():
            number_of_certain_samples = certain_apis.get(api_fam, {}).get(endpoint, 0)
            rows.append([endpoint, api_fam, category, count, number_of_certain_samples])

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

# Paths to the input files
api_categories_file = 'dataset/toolbench/api_categories.txt'
api_report_file = 'dataset/toolbench/endpoint_report.txt'
output_csv_file = 'dataset/toolbench/full_report.csv'

# Parse the input files
api_categories = parse_api_categories(api_categories_file)
certain_apis, all_apis = parse_api_report(api_report_file)

# Generate the output CSV
generate_csv(output_csv_file, api_categories, certain_apis, all_apis)

print(f"CSV file '{output_csv_file}' generated successfully.")
