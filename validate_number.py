import re
from datetime import datetime
import string

# Define dictionaries for character-to-integer and integer-to-character mappings
dict_char_to_int = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'T': '1', 'Z': '2', 'J': '3', 'A': '4', 'S': '5', 'G': '6', 'K': '7',
                    'B': '8'}
dict_int_to_char = {'0': 'Q', '1': 'T', '2': 'Z', '3': 'J', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'}
dict_no_change = {'O': '0', 'I': '1'}


def preprocess_plate_number(plate_number):
    """
    Preprocess the plate number to fix common OCR errors based on Indian number plate format.
    Corrects characters based on their expected position (letters vs numbers).
    Supports both single-line and two-line (two-wheeler) formats.
    """
    # Handle two-line format (common for two-wheelers)
    if '\n' in plate_number or len(plate_number.split()) == 2:
        return preprocess_two_line_plate(plate_number)
    
    # Remove spaces and convert to uppercase
    cleaned_plate = plate_number.replace(" ", "").upper()
    
    # Detect plate format and apply corrections accordingly
    if len(cleaned_plate) >= 4 and cleaned_plate[2:4] == "VA":
        return preprocess_vintage_series(cleaned_plate)
    elif len(cleaned_plate) >= 4 and cleaned_plate[2:4] == "BH":
        return preprocess_bharat_series(cleaned_plate)
    else:
        return preprocess_standard_series(cleaned_plate)


def preprocess_standard_series(plate_number):
    """
    Preprocess standard HSRP format: [State(2)][District(2)][Series(1-3)][Number(4)]
    Expected: DL01AB1234
    """
    if len(plate_number) < 7:  # Minimum valid length
        return plate_number
    
    corrected = list(plate_number)
    
    # State code (positions 0-1): Should be letters
    for i in range(2):
        if i < len(corrected):
            char = corrected[i]
            if char.isdigit():
                # Convert digit to letter if mistakenly recognized
                if char in dict_int_to_char:
                    corrected[i] = dict_int_to_char[char]
            elif char in dict_no_change:
                corrected[i] = dict_no_change[char]
    
    # District code (positions 2-3): Should be digits
    for i in range(2, 4):
        if i < len(corrected):
            char = corrected[i]
            if char.isalpha():
                # Convert letter to digit if mistakenly recognized
                if char in dict_char_to_int:
                    corrected[i] = dict_char_to_int[char]
            elif char in dict_no_change:
                corrected[i] = dict_no_change[char]
    
    # Series (positions 4 to -4): Should be letters (1-3 characters)
    series_end = len(corrected) - 4
    for i in range(4, series_end):
        if i < len(corrected):
            char = corrected[i]
            if char.isdigit():
                # Convert digit to letter if mistakenly recognized
                if char in dict_int_to_char:
                    corrected[i] = dict_int_to_char[char]
            elif char in dict_no_change:
                corrected[i] = dict_no_change[char]
    
    # Number (last 4 positions): Should be digits
    for i in range(len(corrected) - 4, len(corrected)):
        if i >= 0 and i < len(corrected):
            char = corrected[i]
            if char.isalpha():
                # Convert letter to digit if mistakenly recognized
                if char in dict_char_to_int:
                    corrected[i] = dict_char_to_int[char]
            elif char in dict_no_change:
                corrected[i] = dict_no_change[char]
    
    return ''.join(corrected)


def preprocess_two_line_plate(plate_number):
    """
    Preprocess two-line HSRP format (common for two-wheelers):
    Line 1: [State(2)][District(2)]
    Line 2: [Series(1-3)][Number(4)]
    Expected: "DL01\nA1234" or "DL01 A1234"
    """
    # Split by newline or space
    if '\n' in plate_number:
        lines = plate_number.split('\n')
    else:
        lines = plate_number.split()
    
    if len(lines) != 2:
        # If not exactly 2 parts, treat as single line
        return preprocess_standard_series(plate_number.replace('\n', '').replace(' ', '').upper())
    
    # Clean up each line
    line1 = lines[0].replace(' ', '').upper().strip()
    line2 = lines[1].replace(' ', '').upper().strip()
    
    # Validate line lengths
    if len(line1) < 3 or len(line1) > 4 or len(line2) < 4 or len(line2) > 7:
        # Invalid format, try as single line
        combined = line1 + line2
        return preprocess_standard_series(combined)
    
    # Process line 1 (state + district)
    corrected_line1 = list(line1)
    
    # State code (positions 0-1): Should be letters
    for i in range(min(2, len(corrected_line1))):
        char = corrected_line1[i]
        if char.isdigit():
            if char in dict_int_to_char:
                corrected_line1[i] = dict_int_to_char[char]
        elif char in dict_no_change:
            corrected_line1[i] = dict_no_change[char]
    
    # District code (positions 2-3): Should be digits
    for i in range(2, min(4, len(corrected_line1))):
        char = corrected_line1[i]
        if char.isalpha():
            if char in dict_char_to_int:
                corrected_line1[i] = dict_char_to_int[char]
        elif char in dict_no_change:
            corrected_line1[i] = dict_no_change[char]
    
    # Process line 2 (series + number)
    corrected_line2 = list(line2)
    
    # Determine where series ends and number begins (last 4 should be digits)
    series_end = len(corrected_line2) - 4
    
    # Series part: Should be letters
    for i in range(min(series_end, len(corrected_line2))):
        char = corrected_line2[i]
        if char.isdigit():
            if char in dict_int_to_char:
                corrected_line2[i] = dict_int_to_char[char]
        elif char in dict_no_change:
            corrected_line2[i] = dict_no_change[char]
    
    # Number part (last 4): Should be digits
    for i in range(max(0, series_end), len(corrected_line2)):
        char = corrected_line2[i]
        if char.isalpha():
            if char in dict_char_to_int:
                corrected_line2[i] = dict_char_to_int[char]
        elif char in dict_no_change:
            corrected_line2[i] = dict_no_change[char]
    
    # Combine the corrected lines
    corrected_plate = ''.join(corrected_line1) + ''.join(corrected_line2)
    return corrected_plate


def preprocess_vintage_series(plate_number):
    """
    Preprocess vintage HSRP format: [State(2)]VA[Series(1-3)][Number(4)]
    Expected: DLVA1234 or DLVAAB1234
    """
    if len(plate_number) < 8:  # Minimum valid length
        return plate_number
    
    corrected = list(plate_number)
    
    # State code (positions 0-1): Should be letters
    for i in range(2):
        if i < len(corrected):
            char = corrected[i]
            if char.isdigit():
                if char in dict_int_to_char:
                    corrected[i] = dict_int_to_char[char]
            elif char in dict_no_change:
                corrected[i] = dict_no_change[char]
    
    # VA is fixed (positions 2-3)
    
    # Series (positions 4 to -4): Should be letters
    series_end = len(corrected) - 4
    for i in range(4, series_end):
        if i < len(corrected):
            char = corrected[i]
            if char.isdigit():
                if char in dict_int_to_char:
                    corrected[i] = dict_int_to_char[char]
            elif char in dict_no_change:
                corrected[i] = dict_no_change[char]
    
    # Number (last 4 positions): Should be digits
    for i in range(len(corrected) - 4, len(corrected)):
        if i >= 0 and i < len(corrected):
            char = corrected[i]
            if char.isalpha():
                if char in dict_char_to_int:
                    corrected[i] = dict_char_to_int[char]
            elif char in dict_no_change:
                corrected[i] = dict_no_change[char]
    
    return ''.join(corrected)


def preprocess_bharat_series(plate_number):
    """
    Preprocess Bharat HSRP format: [Year(2)]BH[Number(4)][Series(1-3)]
    Expected: 22BH1234AB
    """
    if len(plate_number) < 9:  # Minimum valid length
        return plate_number
    
    corrected = list(plate_number)
    
    # Year (positions 0-1): Should be digits
    for i in range(2):
        if i < len(corrected):
            char = corrected[i]
            if char.isalpha():
                if char in dict_char_to_int:
                    corrected[i] = dict_char_to_int[char]
            elif char in dict_no_change:
                corrected[i] = dict_no_change[char]
    
    # BH is fixed (positions 2-3)
    
    # Number (positions 4-7): Should be digits
    for i in range(4, 8):
        if i < len(corrected):
            char = corrected[i]
            if char.isalpha():
                if char in dict_char_to_int:
                    corrected[i] = dict_char_to_int[char]
            elif char in dict_no_change:
                corrected[i] = dict_no_change[char]
    
    # Series (positions 8+): Should be letters
    for i in range(8, len(corrected)):
        char = corrected[i]
        if char.isdigit():
            if char in dict_int_to_char:
                corrected[i] = dict_int_to_char[char]
        elif char in dict_no_change:
            corrected[i] = dict_no_change[char]
    
    return ''.join(corrected)


def validate_and_format_plate(plate_number):
    """
    Preprocess the plate number to fix OCR errors, then validate it.
    Returns the corrected plate number and validation status.
    """

    corrected_plate = preprocess_plate_number(plate_number)
    is_valid, message = validate_hsrp(corrected_plate)
    return corrected_plate, is_valid, message

# Define valid characters for license plates
plateChars = "0123456789ABCDEFGHJKLMNOPQRSTUVWXYZ"

# Define the regular expression pattern for Indian vehicle registration plates
# plate_pattern = r'^[A-Z]{2}[01-99]{2}[ABCDEFGHJKLMNPQRSTUVWXYZ]{0,3}[0-9]{4}$'

# Define valid state codes
valid_state_codes = (
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DN", "DL", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TG", "TS", "TN", "TR",
    "UK", "UA", "UP", "WB"
)


def validate_hsrp(plate_number):

    # Check for special series
    if len(plate_number) >= 4 and "VA" == plate_number[2:4]:
        return validate_vintage_series(plate_number)
    elif len(plate_number) >= 4 and "BH" == plate_number[2:4]:
        return validate_bharat_series(plate_number)
    # elif any(x in plate_number for x in ["CD", "CC", "UN"]):
    #     return validate_diplomatic_series(plate_number)

    # Minimum length check for standard format
    if len(plate_number) < 7:  # At least state(2) + district(1) + number(4)
        return False, "Plate number too short"

    # Extract components for standard format
    state_code = plate_number[:2]
    
    # Handle single or double digit district code
    # Try to identify where district code ends by checking for first non-digit after position 2
    district_end = 2
    while district_end < len(plate_number) and district_end < 4 and plate_number[district_end].isdigit():
        district_end += 1
    
    # If no digits found after state code, it's invalid
    if district_end == 2:
        return False, "Invalid district code"
    
    district_code = plate_number[2:district_end]
    series = plate_number[district_end:-4]
    number = plate_number[-4:]

    if state_code not in valid_state_codes:
        return False, "Invalid state code"

    # Validate district code (should be 1-2 digit number)
    if not district_code.isdigit() or len(district_code) < 1 or len(district_code) > 2:
        return False, "Invalid district code"

    # Validate series (0 to 3 letters, can be empty)
    if len(series) > 3 or (len(series) > 0 and (not series.isalpha() or 'O' in series or 'I' in series)):
        return False, "Invalid series"

    # Validate number (should be 4 digits)
    if not number.isdigit() or len(number) != 4:
        return False, "Invalid number"

    return True, "Valid standard HSRP"


def validate_vintage_series(plate_number):

    # Extract components for standard format
    state_code = plate_number[:2]
    series = plate_number[4:-4]
    number = plate_number[-4:]

    if state_code not in valid_state_codes:
        return False, "Invalid state code"

    # Validate series (1 to 3 letters)
    if not series.isalpha() or len(series) > 3 or 'O' in series or 'I' in series:
        return False, "Invalid series"

    # Validate number (should be 4 digits)
    if not number.isdigit() or len(number) != 4:
        return False, "Invalid number"

    return True, "Valid Vintage series HSRP"


def validate_bharat_series(plate_number):

    # Extract components for standard format
    reg_year = plate_number[:2]
    number = plate_number[4:8]
    series = plate_number[8:]

    current_year = datetime.now().year % 100
    if 21 > int(reg_year) or int(reg_year) > current_year:
        return False, "Invalid registration year in BH Series"

    # Validate number (should be 4 digits)
    if not number.isdigit() or len(number) != 4:
        return False, "Invalid number in BH Series"

    # Validate series (1 to 3 letters)
    if not series.isalpha() or len(series) > 3 or 'O' in series or 'I' in series:
        return False, "Invalid series in BH Series"

    return True, "Valid Bharat series HSRP"


# def validate_diplomatic_series(plate_number):
#     if not re.match(r'^[0-9]{3}(CD|CC|UN)[0-9]{1,4}$', plate_number):
#         return False, "Invalid Diplomatic series format"
#     return True, "Valid Diplomatic series HSRP"


# def validate_and_format_plate(number):
#     """
#     Validate the license plate and format it if possible.
#     """
#     it_is_valid, msg = validate_hsrp(number)
#     if it_is_valid:
#         return number, msg
#     else:
#         return number, msg


# Test the function
test_plates = [
    "DL01VA1234",  # Valid standard
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
    "HR0ZAB0123",
    "PBO2DB7188",
    "PB08C4505]",
    "PBQBE48032"
]

# Example usage with OCR errors
if __name__ == "__main__":
    # Test cases with common OCR errors
    ocr_test_plates = [
        "DLO1AB1234",    # O instead of 0 in district code
        "0L01AB1234",    # 0 instead of D in state code  
        "DL01A81234",    # 8 instead of B in series
        "DL01AB123A",    # A instead of 4 in number
        "MH O2 CD 5678", # O instead of 0 with spaces
        "22BHA234AB",    # A instead of 1 in Bharat series number
        "DLVA A8I234",   # 8 instead of B, I instead of 1 in vintage
        "PB02D8718B",    # 8 instead of B in series
    ]
    
    print("Testing OCR Error Correction:")
    print("=" * 50)
    
    for plate in ocr_test_plates:
        corrected_plate, is_valid, message = validate_and_format_plate(plate)
        print(f"Original: {plate}")
        print(f"Corrected: {corrected_plate}")
        print(f"Status: {message}")
        print("-" * 30)

# Uncomment to run tests
# for plate in test_plates:
#     result, status = validate_and_format_plate(plate)
#     print(f"Original: {plate}, Result: {result}, Status: {status}")
