import pandas as pd
from fpdf import FPDF
from datetime import datetime

# Set the file paths
csv_file_path = '/Users/nemo/Desktop/vlm_for_IoT/images/captions.csv'
image_folder_path = '/Users/nemo/Desktop/vlm_for_IoT/images/'
highlight_keywords = ["man", "car", "human", "person", "girl", "woman", "vehicle", "child", "dog", "cat", "bicycle", "motorcycle"]

# Load the CSV data
df = pd.read_csv(csv_file_path)

# Get the current date for the subheader
current_date = datetime.now().strftime("%Y-%m-%d")

# Initialize PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Report title
pdf.set_font("Arial", size=24)
pdf.cell(0, 10, txt="Surveillance Summary Report", ln=True, align='C')
pdf.ln(10)

# Subheader with date
pdf.set_font("Arial", size=16)
pdf.cell(0, 10, txt=f"Date: {current_date}", ln=True, align='C')
pdf.ln(20)

# Loop through each row and add details to the report
for index, row in df.iterrows():
    sequence_no = row['Sequence No']
    timestamp = row['Timestamp']
    caption = row['Caption']

    # Get image path and dimensions
    image_path = f"{image_folder_path}{sequence_no}.jpg"
    img_width, img_height = 100, 75  # Fixed dimensions to avoid cutting

    # Check if the next image will fit on the current page, if not, add a new page
    if pdf.get_y() + img_height + 20 > 270:  # Adding space for text below the image
        pdf.add_page()

    # Center align the image
    x_offset = (210 - img_width) / 2  # Calculate x_offset for center alignment (A4 width is 210mm)
    pdf.image(image_path, x=x_offset, y=pdf.get_y(), w=img_width)

    pdf.ln(img_height + 5)  # Move cursor after the image

    # Check for the word "man" and change text color to red if present
    if any(word in caption.lower() for word in highlight_keywords):
        pdf.set_text_color(255, 0, 0)  # Red color
    else:
        pdf.set_text_color(0, 0, 0)  # Black color

    # Add timestamp and caption to the PDF below the image with center alignment
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"Time: {timestamp}", ln=True, align='C')
    pdf.multi_cell(0, 10, txt=f"Event: {caption}", align='C')

    pdf.ln(10)  # Add some space after the caption

    # Reset text color to black after caption
    pdf.set_text_color(0, 0, 0)

# Save the PDF
output_file = "/Users/nemo/Desktop/vlm_for_IoT/Surveillance_Summary_Report.pdf"
pdf.output(output_file)

print(f"Report generated successfully at {output_file}!")
