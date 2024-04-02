import csv
import os
from PIL import Image

# Define input and output file paths
#input_file = "/mnt/c/Unet/track1/normal/driving_log_Linux.csv" old simulation path
#output_file = "/mnt/c/Unet/track1/normal/driving_log.csv"

def csv_transformer(input_file, output_file):
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
            updated_row = [col.replace("/home/linfeng/Pictures/IMG", "/mnt/c/Unet/track1/Lake_Day_Sun/IMG").replace('.png', '.jpg') if i < 3 else col for
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



def remove_last_three(csv_path, output_path):
    with open(csv_path, 'r') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        data = [header]  # Keep the header

        for row in reader:
            data.append(row[:-3])

    with open(output_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(data)

if __name__ == '__main__':
    input_path_csv = "/mnt/c/Unet/track1/Lake_Day_Sun/driving_log_Linux.csv"
    output_path_csv = "/mnt/c/Unet/track1/Lake_Day_Sun/driving_log.csv"
    input_folder = "/mnt/c/Unet/track1/Lake_Day_Sun/IMG"
    #convert_png_to_jpg(input_folder)
    #csv_transformer(input_path_csv,output_path_csv)
    remove_last_three(output_path_csv,output_path_csv)