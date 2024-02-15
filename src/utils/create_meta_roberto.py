import csv

# Open the text file for reading
with open('/home/ai2lab/datasets/roberto/test.txt') as f:
    # Read all lines
    lines = f.readlines()

# Open a CSV file for writing
with open('/home/ai2lab/datasets/roberto/test.csv', 'w') as csvfile:
    # Create CSV writer
    writer = csv.writer(csvfile)

    # Iterate through the lines
    for line in lines:
        # Split on space
        parts = line.strip().split(' ')
        parts[0] = parts[0].replace(".nii.gz", "")
        # Write row to CSV
        writer.writerow(parts)

print('Text file converted to CSV successfully!')