import os
import cv2
import pytesseract
import np
import re  # regex matching
import json
from nltk.tag.stanford import StanfordNERTagger

EMAIL_REGEX = r"[^@]+@[^@]+\.[^@]+"
PHONE_REGEX = r"^(\(?\+?[0-9]*\)?)?[0-9_\- \(\)]*$"
#ADDRESS_REGEX = r"\d{1,3}.?\d{0,3}\s[a-zA-Z]{2,30}\s[a-zA-Z]{2,15}"

output_dir = './output'
st = StanfordNERTagger(
        r'stanford-ner\classifiers\english.all.3class.distsim.crf.ser.gz',
        'stanford-ner\stanford-ner.jar')


def is_text_match_regex(text, regex):
    is_valid = regex.match(text)
    return is_valid is not None


def is_email_valid(text):
    return is_text_match_regex(text, re.compile(EMAIL_REGEX))


def is_phone_valid(text):
    return is_text_match_regex(text, re.compile(PHONE_REGEX))


def get_data_from_text(text):
    for line in text.splitlines():
        for substring in line.split(' '):
            if is_email_valid(substring):
                email = substring
            elif is_phone_valid(substring):
                phone = substring
    return email, phone


def get_string(img_path):
    # Read image using opencv
    img = cv2.imread(img_path)

    # Extract the file name without the file extension
    file_name = os.path.basename(img_path).split('.')[0]
    file_name = file_name.split()[0]

    # Create a directory for outputs
    output_path = os.path.join(output_dir, file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Rescale the image, if needed.
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    # Apply blur to smooth out the edges
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply threshold to get image with only b&w (binarization)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Save the filtered image in the output directory
    save_path = os.path.join(output_path, file_name + "_filter_" + "binarization" + ".jpg")
    cv2.imwrite(save_path, img)

    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(img, lang="eng")
    return result


def get_json_card_data(image_path):
    result = get_string(image_path)
    email, phone = get_data_from_text(result)
    person_name = get_full_name(result)
    company = ""
    address = ""
    website = ""
    data = {"personName": person_name, "phoneNumber": phone, "company": company, "address": address,
            "email": email,  "website": website}
    json_data = json.dumps(data)
    return json_data


def get_full_name(text):
    r = st.tag(text.split())
    full_name = ""
    from itertools import groupby
    for tag, chunk in groupby(r, lambda x: x[1]):
        if tag != "O":
            full_name = " ".join(w for w, t in chunk)
    if len(full_name.split(' ')) >= 2:
        return full_name.split(' ')[0] + " " + full_name.split(' ')[1]
    return full_name


if __name__ == "__main__":
    image_path = r'prototype_cards\card2.jpg'
    json_data = get_json_card_data(image_path)
    print(json_data)
