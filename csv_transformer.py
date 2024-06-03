import csv
import os
from PIL import Image


def csv_transformer_png(input_file, output_file):
    # Define the column headers

    column_headers = ["center,left,right,steering,throttle,brake,speed"]

    # Open the input and output files
    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        # Write the column headers to the output CSV file
        writer.writerow(column_headers)

        # Skip the header row in the input CSV file
        next(reader)

        # Iterate through each row in the input CSV file
        for row in reader:
            # Update file paths in the first three columns
            updated_row = [
                col.replace("C:\\Unet\\track2\\normal2\\IMG\\", "/mnt/c/Unet/track2/normal2/IMG/") if i < 3 else col for
                i, col
                in enumerate(row)]
            # Write the updated row to the output CSV file
            writer.writerow(updated_row)


def remove_last_four(csv_path, output_path):
    with open(csv_path, 'r') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        data = [header]  # Keep the header

        for row in reader:
            data.append(row[:-4])


    with open(output_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(data)

def transform_comma2_period(input_csv, output_csv):
    with open(input_csv, 'r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        data = [header]

        for row in reader:
            num_elements = len(row)
            print(num_elements)


            last_two_elements = '.'.join(row[-2:])
            fourth_and_fifth_elements = '.'.join(row[3:5])

            if num_elements == 10:
                sixth_and_seventh_elements = '.'.join(row[5:7])
                modified_row = row[0:3] + [fourth_and_fifth_elements] + [sixth_and_seventh_elements] + row[7:8] + [last_two_elements]
            elif num_elements == 9:
                modified_row = row[0:3] + [fourth_and_fifth_elements] + row[5:-2] + [last_two_elements]
            elif num_elements == 8:
                modified_row = row[0:-2] + [last_two_elements]

            data.append(modified_row)

    # Write the data to the output CSV file
    with open(output_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(data)


def transform(input_path, output_path):
    csv_transformer_png(input_path, output_path)
    remove_last_four(output_path, output_path)
    transform_comma2_period(output_path,output_path)

    print("finish transform")
'''
def csv_transformer_jpg(input_file, output_file):
    # Define the column headers

    column_headers = ["center", "left", "right", "steering", "throttle", "brake", "speed"]

    # Open the input and output files
    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        # Write the column headers to the output CSV file
        writer.writerow(column_headers)

        # Skip the header row in the input CSV file
        next(reader)

        # Iterate through each row in the input CSV file
        for row in reader:
            # Update file paths in the first three columns
            updated_row = [
                col.replace("C:/Unet/track1/normal/IMG", "/mnt/c/Unet/track1/Lake_Day_Sun/IMG").replace('.png',
                                                                                                         '.jpg') if i < 3 else col
                for
                i, col
                in enumerate(row)]
            # Write the updated row to the output CSV file
            writer.writerow(updated_row)
            
def convert_png_to_jpg(input_folder):
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    # Create an output folder for JPG images
    output_folder = os.path.join(input_folder, "jpg_images")
    os.makedirs(output_folder, exist_ok=True)

    # Loop through each file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".png"):
            # Open the PNG image
            png_image_path = os.path.join(input_folder, file_name)
            image = Image.open(png_image_path)

            # Create the output file path for the JPG image
            jpg_image_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".jpg")

            # Convert and save the image as JPG
            image.convert("RGB").save(jpg_image_path)
            print(f"Converted '{file_name}' to '{os.path.basename(jpg_image_path)}'")

    print("Conversion complete.")

'''
def compress(input_file, output_file):

    with open(input_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        data_array = [",".join(row) for row in csv_reader]

    # Daten in eine neue CSV-Datei schreiben
    with open(input_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for data in data_array:
            csv_writer.writerow([data])

    csv_transformer_png(input_file, output_file)


if __name__ == '__main__':
    input_path_csv = "/mnt/c/Unet/track2/normal2/driving_log_Linux.csv"
    output_path_csv = "/mnt/c/Unet/track2/normal2/driving_log.csv"
    #transform(input_path_csv, output_path_csv)
    compress(input_path_csv,output_path_csv)