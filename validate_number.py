import re
from datetime import datetime
import string

# Define dictionaries for character-to-integer and integer-to-character mappings
dict_char_to_int = {'O': '0', 'D': '0', 'Q': '0', 'T': '1', 'Z': '2', 'J': '3', 'A': '4', 'S': '5', 'G': '6', 'K': '7',
                    'B': '8'}
dict_int_to_char = {'0': 'Q', '1': 'T', '2': 'Z', '3': 'J', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'}
dict_no_change = {'O': '0', 'I': '1'}

# Define valid characters for license plates
plateChars = "0123456789ABCDEFGHJKLMNOPQRSTUVWXYZ"

# Define the regular expression pattern for Indian vehicle registration plates
plate_pattern = r'^[A-Z]{2}[0-9OI]{1,2}[ABCDEFGHJKLMNPQRSTUVWXYZ]{0,3}[0-9]{4}$'

# Define valid state codes
state_codes = (
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DN", "DL", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TG", "TS", "TN", "TR",
    "UK", "UA", "UP", "WB"
)


def validate_hsrp(plate_number):
    # Regular expression pattern for HSRP
    pattern = r'^([A-Z]{2}\s?[0-9]{1,2}\s?[A-Z]{1,3}\s?[0-9]{4})$|^([0-9]{2}\s?BH\s?[0-9]{4}\s?[A-Z]{1,2})$|^(([A-Z]{2}\s?VA\s?[A-Z]{2}\s?[0-9]{4})|([0-9]{3}\s?(CD|CC|UN)\s?[0-9]{1,4}))$'

    # Check if the plate number matches the general pattern
    if not re.match(pattern, plate_number):
        return False, "Invalid format"

    # Remove spaces from the plate number
    plate_number = plate_number.replace(" ", "")

    # Check for special series
    if plate_number.startswith("VA"):
        return validate_vintage_series(plate_number)
    elif "BH" in plate_number:
        return validate_bharat_series(plate_number)
    elif any(x in plate_number for x in ["CD", "CC", "UN"]):
        return validate_diplomatic_series(plate_number)

    # Extract components for standard format
    state_code = plate_number[:2]
    district_code = plate_number[2:4]
    series = plate_number[4:-4]
    number = plate_number[-4:]

    # Validate state code
    valid_state_codes = [
        "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
        "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
        "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP",
        "WB", "TG"
    ]
    if state_code not in valid_state_codes:
        return False, "Invalid state code"

    # Validate district code (should be a two-digit number)
    if not district_code.isdigit() or len(district_code) != 2:
        return False, "Invalid district code"

    # Validate series (1 to 3 letters)
    if not series.isalpha() or len(series) > 3 or 'O' in series or 'I' in series:
        return False, "Invalid series"

    # Validate number (should be 4 digits)
    if not number.isdigit() or len(number) != 4:
        return False, "Invalid number"

    return True, "Valid standard HSRP"


def validate_vintage_series(plate_number):
    if not re.match(r'^[A-Z]{2}VA[A-Z]{2}[0-9]{4}$', plate_number):
        return False, "Invalid Vintage series format"
    return True, "Valid Vintage series HSRP"


def validate_bharat_series(plate_number):
    if not re.match(r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$', plate_number):
        return False, "Invalid Bharat series format"
    year = int(plate_number[:2])
    current_year = datetime.now().year % 100
    if year > current_year:
        return False, "Invalid year in Bharat series"
    return True, "Valid Bharat series HSRP"


def validate_diplomatic_series(plate_number):
    if not re.match(r'^[0-9]{3}(CD|CC|UN)[0-9]{1,4}$', plate_number):
        return False, "Invalid Diplomatic series format"
    return True, "Valid Diplomatic series HSRP"


def format_license_plate(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.
    """
    formatted_plate = ''
    for i, char in enumerate(text):
        if i < 2:
            formatted_plate += char
        elif char in dict_char_to_int:
            formatted_plate += dict_char_to_int[char]
        elif char in dict_int_to_char:
            formatted_plate += dict_int_to_char[char]
        else:
            formatted_plate += char
    return formatted_plate


def validate_and_format_plate(number):
    """
    Validate the license plate and format it if possible.
    """
    it_is_valid, _ = validate_hsrp(number)
    if it_is_valid:
        return number, "Valid"
    else:
        formatted_plate = format_license_plate(number)
        now_is_valid, _ = validate_hsrp(formatted_plate)
        if now_is_valid:
            return formatted_plate, "Formatted"
        else:
            return number, "Invalid"


# Test the function
test_plates = [
    "DL01AB1234",  # Valid standard
    "MH 02 CD 5678",  # Valid standard with spaces
    "KA03EF9012",  # Valid standard
    "UP 16 BH 3456",  # Valid standard (BH series)
    "TN 01 G 1234",  # Valid (TN Government vehicle)
    "TN 58 N 4006",  # Valid (TN Government Transport Bus)
    "AP 40 Z 5678",  # Valid (AP State Road Transport bus)
    "ML 01 AB 9999",  # Valid (ML Government vehicle)
    "GJ 01 A 0001",  # Valid standard
    "22BH1234AB",  # Valid Bharat series
    "DLVA AB1234",  # Valid Vintage series
    "123CD1234",  # Valid Diplomatic series
    "XX99YY9999",  # Invalid state code
    "DL1A1234",  # Invalid format
    "MH02CD123",  # Invalid number (less than 4 digits)
    "KA03EFG9012",  # Invalid series (more than 3 letters)
    "UP 16 B3 3456",  # Invalid series (contains number)
    "TN 01 1234",  # Invalid (missing series)
    "99BH1234AB",  # Invalid Bharat series (year)
    "ANVA ABC123",  # Invalid Vintage series
    "1234CD1234",  # Invalid Diplomatic series
    "DL01AB1234",
    "MH02CD5678",
    "KA03EF9012",
    "UP80GH3456",
    "TN07IJ7890",
    "GJ18KL2345",
    "INVALID123",
    "AP09MN6789",
    "WB10OP0123",
    "RJ14QR4567",
    "HR0ZAB0123"
]

# for plate in test_plates:
#     result, status = validate_and_format_plate(plate)
#     print(f"Original: {plate}, Result: {result}, Status: {status}")


for plate in test_plates:
    is_valid, message = validate_and_format_plate(plate)
    print(f"{plate}: {message}")
