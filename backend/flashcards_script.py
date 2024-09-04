# Script to test the flashcards API without running the server

from flashcards import generate_flashcards, FlashCard
import logging
import argparse
import pdfplumber
from dotenv import load_dotenv
import os

load_dotenv()

TEST_DIR = 'test_data/'
DEFAULT_OUT_FILE = 'test_output.txt'
DEFAULT_LOG_FILE = 'test_logs.txt'

parser = argparse.ArgumentParser()

parser.add_argument('--input', required=True, help='Input file name')
parser.add_argument('--out', required=False, help='Output file name')
parser.add_argument('--log', required=False, help='Log file name')
args = parser.parse_args()

in_file_name = args.input
out_file_name = TEST_DIR + (args.out if args.out is not None else DEFAULT_OUT_FILE)
log_file_name = TEST_DIR + (args.log if args.log is not None else DEFAULT_LOG_FILE)

if not os.path.exists(log_file_name):
    open(log_file_name, 'w').close()
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    filename=log_file_name,
)

input_text = ""
with pdfplumber.open(in_file_name) as pdf:
    for page in pdf.pages:
        input_text += page.extract_text()


flashcards = generate_flashcards(input_text)
logging.info(f"Generated {len(flashcards)} flashcards")

# Check if the file exists, if not create it
if not os.path.exists(out_file_name):
    open(out_file_name, 'w').close()

with open(out_file_name, 'w') as out_file:
    out_file.write("\n".join([str(f) for f in flashcards]))
