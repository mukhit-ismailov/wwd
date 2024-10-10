import pytesseract
from PIL import Image
import numpy as np
from pytesseract import Output
import pandas as pd
import pypdfium2 as pdfium


def main():
    s3_bucket = 'vantage-data-sample-tmp'
    src_folder = 'OCR/input/'

    ocr_extract = OcrExtract(s3_bucket, src_folder)
    df = ocr_extract.process_images()
    df = ocr_extract.summarize_text(df)
    api.save(df)


class OcrExtract:

    def __init__(self, s3_bucket, src_folder):
        self.src_folder = src_folder
        self.s3_bucket = s3_bucket

    def convert_pdf2image(self):
        filenames = []
        item_list = api.get_files_from_s3(self.s3_bucket, self.src_folder)
        for item in item_list:
            if str(item.key).endswith('pdf'):
                filename = api.load_file_from_s3(item)
                pdf = pdfium.PdfDocument(filename)
                for i in range(len(pdf)):
                    page = pdf[i]
                    image = page.render(scale=4).to_pil()
                    file_name = filename.split("/")[-1]
                    file_name = f"{file_name}_page{i+1:03d}.jpg"
                    image.save(file_name)
                    filenames.append(file_name)
        return filenames

    def get_filenames(self) -> list:
        try:
            item_list = api.get_files_from_s3(self.s3_bucket, self.src_folder)
            filenames = []
            for item in item_list:
                filename = api.load_file_from_s3(item)
                if not filename.endswith('/') and not filename.endswith('__') and not filename.endswith('pdf'):
                    filenames.append(filename)
            if not filenames:
                LOG.exception(f"Caught an exception during get_filenames() - no files found in {self.src_folder}")
            return filenames
        except Exception as e:
            LOG.exception(f"Caught an exception during get_filenames() - {e}")

    def get_image_data(self, filename) -> pd.DataFrame:
        try:
            return pytesseract.image_to_data(Image.open(filename),lang = 'eng+ara', output_type=Output.DATAFRAME)
        except Exception as e:
            LOG.exception(f"Caught an exception during get_image_data() - {e}")

    def create_image_segment_tesseract(self, image_data, index, text_index, texts) -> dict:
        try:
            xpos = image_data['left'][index]
            ypos = image_data['top'][index]
            width = image_data['width'][index]
            height = image_data['height'][index]
            shape = "RECTANGLE"
            text = texts[text_index]
            coordinates = [xpos, ypos, xpos + width, ypos, xpos + width, ypos + height, xpos, ypos + height]

            return {
                "coordinates": coordinates,
                "shape": shape,
                "object_label": None,
                "text": text
            }
        except Exception as e:
            LOG.exception(f"Caught an exception during create_image_segment_tesseract() - {e}")

    def populate_text(self, image_data, getting_sentences_id) -> list:
        try:
            text_result = []
            for index in getting_sentences_id[0]:
                text_result.append(str(image_data['text'][index]))
            return text_result
        except Exception as e:
            LOG.exception(f"Caught an exception during populate_text() - {e}")


    def process_ocr_boxes(self, image_data) -> list:
        try:
            convert_d = np.asarray(image_data['level'])
            getting_sentences_id = np.where(convert_d == 5)
            texts = self.populate_text(image_data, getting_sentences_id)
            image_segments = []
            for text_index, index in enumerate(getting_sentences_id[0]):
                if len(texts[text_index].strip()) > 0:
                    image_segments.append(self.create_image_segment_tesseract(image_data, index, text_index, texts))
            return image_segments, texts
        except Exception as e:
            LOG.exception(f"Caught an exception during process_ocr_boxes() - {e}")


    def process_images(self, return_with_boxes=False) -> pd.DataFrame:
        try:
            filenames_pdf = self.convert_pdf2image()
            filenames_image = self.get_filenames()
            filenames = filenames_pdf + filenames_image
            data = []  # Initialize an empty list
            for filename in filenames:
                file_name = filename.split("/")[-1]
                image_data = self.get_image_data(filename)
                output, text_output = self.process_ocr_boxes(image_data)
                print(f"----{file_name}----")
                print(" ".join(text_output))
                # Determine if the file is a PDF or an image based on the extension
                pdf_or_image = 'pdf' if ".pdf" in filename else 'image'
                # Create a dictionary and append it to the list
                data.append({
                    'filename': file_name,
                    'pdf_or_image': pdf_or_image,
                    'text_output': " ".join(text_output)
                })
            # Convert the list of dictionaries into a DataFrame
            df_combined = pd.DataFrame(data)
            return df_combined
        except Exception as e:
            LOG.exception(f"Caught an exception during process_images() - {e}")

    def summarize_text(self, df):
        df['text_summarized'] = df['text_output'].apply(self.summarize_long_text)
        return df

    def summarize_long_text(self, text):
        import torch
        from transformers import pipeline
        batch_size = 2048
        device = 0 if torch.cuda.is_available() else -1
        chunks = [text[i:i + batch_size] for i in range(0, len(text), batch_size)]
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
        summaries = [summarizer(chunk, max_length=130, min_length=30, do_sample=False, truncation=True)[0]['summary_text'] for chunk in chunks]
        for i, summary in enumerate(summaries):
            print(f"Summary: {summary}\n\n")
        return ' \n\n'.join(summaries)


main()
