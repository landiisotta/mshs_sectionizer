from typing import List, Iterator, Union, Tuple
import pandas as pd
import spacy
from spacy.tokens import Doc
import os
import datetime
import json

MIN_LENGTH = 150
HPI = 'history_of_present_illness'
SECTIONIZER_CONFIG = json.load(open('sectionizer_config.json', 'r'))


def sectionize(headers_path: str, text_list: List[str]) -> Iterator[Union[Doc, Doc, Doc]]:
    """
    Takes as input a json file with headers and a list of clinical notes
    and runs medspacy sectionizer. Return a generator of Spacy language model objects
    :param headers_path: path to the json file in MedSpacy format
    :param text_list: list of string
    :return: generator of Spacy language model objects
    """
    nlp = spacy.load('en_core_web_sm')
    SECTIONIZER_CONFIG["rules"] = headers_path
    for name in nlp.pipe_names:
        nlp.disable_pipe(name)
    nlp.add_pipe('medspacy_sectionizer',
                 config=SECTIONIZER_CONFIG)
    assert 'medspacy_sectionizer' in nlp.pipe_names
    docs = nlp.pipe(text_list)
    return docs


def _find_headers(dt_row: Tuple, note_doc: Doc, list_headers: List[str]) -> Tuple:
    dt_with_sections = {}
    if len(set(list_headers).intersection(set(note_doc._.section_categories))) == 0:
        return tuple([dt_row])
    else:
        for section_name, section_text in zip(note_doc._.section_categories, note_doc._.section_spans):
            if section_name in list_headers:
                dt_with_sections.setdefault(section_name, list()).append(section_text.text)
            else:
                if HPI in list_headers and HPI not in note_doc._.section_categories:
                    if section_name == 'note_type' and len(section_text.text.split(' ')) > MIN_LENGTH:
                        dt_with_sections.setdefault(section_name, list()).append(section_text.text)
                    elif section_name is None and len(section_text.text.split(' ')) > MIN_LENGTH:
                        dt_with_sections.setdefault('None', list()).append(section_text.text)
        add_cat = tuple([dt_row + tuple([section_name, ''.join(section_list)]) for section_name, section_list in
                         dt_with_sections.items()])
        return add_cat


def select_headers(dt_path: str, headers_path: str, list_sections: List[str],
                   text_column_name: str = 'Text') -> pd.DataFrame:
    """
    Function that takes as input the path to a dataset and a list of headers to select the sections
    of interest following the rules:
    1) If no headers of interest are found, a file notes_to_check/notes_no_selected_headers.parquet
    with the original dataset and only the notes of interest is dumped. The select_headers() function can then
    be rerun with the notes_no_selected_headers.parquet dataset and
    a different list of headers.
    2) All desired sections are selected, if found. Exceptions apply if `list_sections` contains
    the `history_of_present_illness` category. In that case, sections categorized as `None` or `note_type` are selected
    if longer than `MIN_LENGTH` (empirically determined). This because, in some cases, history of present illness is
    either not found or only contains the header, and the actual patient's history is found at the beginning of the note,
    often sectionized into `note_type` or `None`.

    A dataframe with notes divided into sections is returned
    The results are saved to notes_with_headers.parquet file with sections by row.

    :param dt_path: path to the dataset folder
    :param headers_path: path to json file with headers in MedSpacy format
    :param list_sections: list of sections of interest
    :param text_column_name: string with the name of the dt column containing the note text
    :return: pandas DataFrame
    """
    read_functions = {'csv': pd.read_csv,
                      'xlsx': pd.read_excel,
                      'txt': pd.read_csv,
                      'parquet': pd.read_parquet,
                      'json': pd.read_json}
    file_format = dt_path.split('.')[-1].strip('/')
    notes = read_functions[file_format](dt_path)
    docs = list(sectionize(headers_path=headers_path,
                           text_list=notes[text_column_name]))

    check_docs = []
    dt_sections = pd.DataFrame([], columns=list(notes.columns) + ['section_category', 'section_span'])
    for row, doc in zip(notes.itertuples(), docs):
        add_cat = _find_headers(row, doc, list_sections)
        if len(add_cat[0]) == len(row):
            check_docs.append(add_cat[0])
        else:
            add_cat = pd.DataFrame(add_cat, columns=['Index'] + list(dt_sections.columns)).drop(['Index'], axis=1)
            dt_sections = pd.concat([dt_sections, add_cat])

    date_time = datetime.datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
    if len(check_docs) > 0:
        dt_no_header_to_check = pd.DataFrame(check_docs, columns=['Index'] + list(notes.columns)).drop('Index', axis=1)
        if not os.path.isdir('./notes_to_check'):
            os.makedirs('./notes_to_check')
        dt_no_header_to_check.to_parquet(f'./notes_to_check/notes_no_selected_headers-{date_time}.parquet')
        print(
            f"Dumped {dt_no_header_to_check.shape[0]} notes to folder 'notes_to_check' because no headers from the list were found. Check them manually or discard.")
    dt_sections.to_parquet(f'./notes_with_headers-{date_time}.parquet')
    print(f"Dumped dataset with notes divided into sections.")
    return dt_sections
