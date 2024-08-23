import csv

'''valid_files = [
    "actioncliptest00340", "actioncliptest00396", "actioncliptrain00005", "actioncliptrain00017", "actioncliptrain00029",
    "actioncliptest00341", "actioncliptest00398", "actioncliptrain00006", "actioncliptrain00018", "actioncliptrain00030",
    "actioncliptest00342", "actioncliptest00428", "actioncliptrain00007", "actioncliptrain00019", "actioncliptrain00031",
    "actioncliptest00346", "actioncliptest00456", "actioncliptrain00008", "actioncliptrain00020", "actioncliptrain00032",
    "actioncliptest00351", "actioncliptest00458", "actioncliptrain00009", "actioncliptrain00021", "actioncliptrain00033",
    "actioncliptest00356", "actioncliptest00625", "actioncliptrain00010", "actioncliptrain00022", "actioncliptrain00034",
    "actioncliptest00357", "actioncliptest00633", "actioncliptrain00011", "actioncliptrain00023", "actioncliptrain00036",
    "actioncliptest00358", "actioncliptest00656", "actioncliptrain00012", "actioncliptrain00024", "actioncliptrain00037",
    "actioncliptest00360", "actioncliptrain00001", "actioncliptrain00013", "actioncliptrain00025", "actioncliptrain00038",
    "actioncliptest00362", "actioncliptrain00002", "actioncliptrain00014", "actioncliptrain00026", "actioncliptrain00039",
    "actioncliptest00375", "actioncliptrain00003", "actioncliptrain00015", "actioncliptrain00027", "actioncliptrain00040",
    "actioncliptest00380", "actioncliptrain00004", "actioncliptrain00016", "actioncliptrain00028"
]'''

valid_files = [
    "actioncliptest00340", "actioncliptest00396", "actioncliptrain00005", "actioncliptrain00017", "actioncliptrain00029",
    "actioncliptest00341", "actioncliptest00398", "actioncliptrain00006", "actioncliptrain00018", "actioncliptrain00030",
    "actioncliptest00342", "actioncliptest00428", "actioncliptrain00007", "actioncliptrain00019", "actioncliptrain00031",
    "actioncliptest00346", "actioncliptest00456", "actioncliptrain00008", "actioncliptrain00020", "actioncliptrain00032",
    "actioncliptest00351", "actioncliptest00458", "actioncliptrain00009", "actioncliptrain00021", "actioncliptrain00033",
    "actioncliptest00356", "actioncliptest00625", "actioncliptrain00010", "actioncliptrain00022", "actioncliptrain00034",
    "actioncliptest00357", "actioncliptest00633", "actioncliptrain00011", "actioncliptrain00023", "actioncliptrain00036",
    "actioncliptest00358", "actioncliptest00656", "actioncliptrain00012", "actioncliptrain00024", "actioncliptrain00037",
    "actioncliptest00360", "actioncliptrain00001", "actioncliptrain00013", "actioncliptrain00025", "actioncliptrain00038",
    "actioncliptest00362", "actioncliptrain00002", "actioncliptrain00014", "actioncliptrain00026", "actioncliptrain00039",
    "actioncliptest00375", "actioncliptrain00003", "actioncliptrain00015", "actioncliptrain00027", "actioncliptrain00040",
    "actioncliptest00380", "actioncliptrain00004", "actioncliptrain00016", "actioncliptrain00028"
]


input_csv = './hollywood2/test_original.csv'
output_csv = './hollywood2/test.csv'

with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    header = next(reader)
    writer.writerow(header)

    for row in reader:
        video_name = row[0].strip()
        if video_name in valid_files:
            writer.writerow(row)

print(f"Filtered data saved to {output_csv}")
