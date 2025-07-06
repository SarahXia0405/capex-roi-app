# nlp_utils.py
import re
import spacy
from dateutil import parser as date_parser

import spacy
import subprocess
import importlib.util

# Auto-download spaCy model if not installed
if importlib.util.find_spec("en_core_web_sm") is None:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

nlp = spacy.load("en_core_web_sm")




def parse_user_sentence(sentence):
    doc = nlp(sentence)

    property_id = None
    investment_amount = None
    num_units = None
    capex_date = None
    
    
    # Property ID (e.g. ATL000, BRD123, xyz456 → XYZ456)
    property_id_match = re.search(r"\b([a-zA-Z]{3}\d{3})\b", sentence)
    if property_id_match:
        property_id = property_id_match.group(1).upper()

    # # Property ID (like ATL000 or BRD123)
    # for token in doc:
    #     if token.text.upper().startswith(("ATL", "BRD")):
    #         property_id = token.text.upper()

    # Extract investment amount (e.g. $10000 or 10000)
    money_match = re.search(r"\$?(\d{1,3}(?:,\d{3})+|\d+)", sentence.replace(",", ""))
    if money_match:
        investment_amount = float(money_match.group(1))

    # Extract number of units
    for i, token in enumerate(doc[:-1]):
        if token.like_num and doc[i+1].lemma_ == "unit":
            num_units = int(token.text)

    # Extract capex date (e.g. Jan 2023)
    date_match = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}", sentence, re.IGNORECASE)
    if date_match:
        try:
            parsed_date = date_parser.parse(date_match.group(0))
            capex_date = parsed_date.strftime("%Y-%m-%d")
        except:
            capex_date = None

    return {
        "property_id": property_id,
        "investment_amount": investment_amount,
        "num_units": num_units,
        "capex_date_str": capex_date
    }
