import re
import unicodedata
import json


def preprocess_sections(text: str, header: str, add_sep: str = '  ') -> str:
    """
    According to the Caboodle database schema, clinical notes are stored within the ClinicalNoteTextFact table
    in the Text column, created as the "Concatenation of lines from NOTE_TEXT with spaces between lines" from Clarity.

    Hence, each note line is separated with two blank spaces, i.e., '  '. For this reason, as a general rule,
    sections can be identified as separated with >4 blank spaces, so they can easily be extracted via medspacy
    using a list of ad hoc headers and defining a header in the sectionizer as the presence of the desired
    text preceded by two blank spaces.

    There are cases in which this rule does not apply and users might need to modify the text in order to detect the
    correct sections. In general, adding 2 blank spaces before the header of interest might suffice. For example, this
    sometimes happens for the sections:
    - "ED Assessment/Plan";
    - "Suicide/Violence Risk Assessment";
    - "Violence Risk Assessment  Has the patient experienced";
    - "Suicide Risk Assessment  Suicide Risk Assessment.".

    The "Violence Risk Assessment" section generally starts with the following fixed template:
    "Violence Risk Assessment  Has the patient experienced thoughts of harming others or demonstrated
    aggressive behavior in the last 6 months ? Or has the patient demonstrated any aggressive behavior
    while in the ER? Or Does the patient have any lifetime history of significant aggressive behavior
    (defined as behavior that resulted in serious physical injury, emergency life-saving procedures,
    admission to hospital, or treatment in the ER)?Yes
    If the answer is yes, complete the Violence Risk Assessment.
    If the answer is no, the screen is complete.
    Violence Risk Assessment. (Factors/strengths that can be modified in the short term **). [...]"

    The template at the beginning reports the header name three times so it is not sufficient to insert two blank spaces
    in front of the string because it would end up dividing the section into three. What can be done in such cases
    is to use as header for the insertion of the section string:

    "Violence Risk Assessment  Has the patient experienced"
    Remark: often the Violence Risk Assessment ends up being absent, because the answer to the question above is "No".
    Regardless, the template is always present when the screening is separated from the Suicide Risk Assessment form.

    Similarly happens for the "Suicide Risk Assessment".

    The following function allows to scan the text for the header of interest and then add the desired amount of blanks
    (default:2) in front.

    :param
        text: clinical note text
        header: header text requiring add_sep (e.g., two spaces) in front in order to be correctly identified
        by the medspacy sectionizer
        add_sep: separator text to be inserted before the header of interest. default to two spaces, i.e., '  '
    :return: str clinical note, either as is if header was not found or with add_sep before header
    """
    # Adding the regex for any character except blank space followed by two spaces in front of the
    # header text
    search = re.search(r'[^\s\\]  ' + header, text)
    if search is not None:
        span = search.span()
        return text[:span[0] + 3] + add_sep + text[span[0] + 3:]
    else:
        return text


def uniform_text(text: str) -> str:
    """
    Clinical note text normalization
    :param text: clinical note text
    :return: str clinical note normalized to "NFKC"
    """
    return unicodedata.normalize("NFKC", text)


def text_preprocessing(text: str, headers_file_path: str) -> str:
    """

    :param text:
    :param headers_file_path:
    :return:
    """
    headers_dict = json.load(open(headers_file_path, 'r'))
    mod_text = uniform_text(text)
    for dd in headers_dict['modify_headers']:
        mod_text = preprocess_sections(text=mod_text, header=dd['literal'], add_sep='  ')
    return mod_text
