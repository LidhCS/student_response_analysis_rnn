import itertools as its
import logging

import numpy as np
import pandas as pd

from data_preprocess.constants import (ITEM_IDX_KEY, USER_IDX_KEY,TIME_IDX_KEY, CORRECT_KEY)


LOGGER = logging.getLogger(__name__)

TIME_ID_KEY = 'First Transaction Time'
USER_ID_KEY = 'Anon Student Id'
ORIG_CORRECT_KEY = 'Correct First Attempt'
PROBLEM_NAME = 'Problem Name'
STEP_NAME = 'Step Name'
KC_NAME_STARTS_WITH = 'KC'


def load_data(file_path, item_id_col=PROBLEM_NAME,remove_nan_skill_ids=False, max_interactions_per_user=None,
              drop_duplicates=False, min_interactions_per_user=2):
    """ Load data from KDD Cup data sets.

    :param str file_path: The location of the data
    :param str item_id_col: The column to be used for item_ids in interactions. Likely one of
        PROBLEM_NAME, STEP_NAME, or KC_NAME_STARTS_WITH
    :param str template_id_col: Set a particular column to represent a template id for hierarchical
        IRT. If 'single', assumes a dummy single hierarchical level; if None, no column is retained
        for templates.
    :param str|None concept_id_col: The column to be used for concept_ids in interactions.
        Likely KC_NAME_STARTS_WITH or 'single', in the latter case, all problems are given the same
        concept_id.  If None, no concept column is retained.
    :param bool remove_nan_skill_ids: Whether to remove interactions where the KC column is NaN
    :param int|None max_interactions_per_user: Retain only the first (in time order)
        `max_interactions_per_user` per user. If None, then there is no limit.
    :param bool drop_duplicates: Drop (seemingly) duplicate interactions
    :param int min_interactions_per_user: The minimum number of interactions required to retain
        a user
    :param str|None test_file_path: The KDD Cup data sets break themselves up into a (very large)
        training set and a (very small) test set. This allows you to combine the two files if
        specified. Will be specified in output with an IS_TEST column, which can be used if
        desired by downstream actors.
    :return: processed data, student ids corresponding to the student indices, item ids
        corresponding to the item indices, template ids corresponding to the template indices, and
        concept ids corresponding to the concept indices
    :rtype: (pd.DataFrame, np.ndarray[str], np.ndarray[str], np.ndarray[str])
    """

    data = pd.read_csv(file_path, delimiter='\t')
    # print data.loc[1]
    data_length = len(data)
    LOGGER.info("Read {:3,d} rows from file".format(len(data)))

    LOGGER.info("After test inclusion have {:3,d} rows".format(len(data)))

    data[TIME_IDX_KEY] = np.unique(data[TIME_ID_KEY], return_inverse=True)[1]
    data[CORRECT_KEY] = data[ORIG_CORRECT_KEY] == 1

    # Step names aren't universally unique. Prepend with the problem name to fix this problem.
    data[STEP_NAME] = [':'.join(x) for x in zip(data[PROBLEM_NAME], data[STEP_NAME])]
    kc_name = [column for column in data.columns if column.startswith(KC_NAME_STARTS_WITH)][0]
    if item_id_col and item_id_col.startswith(KC_NAME_STARTS_WITH):
        item_id_col = kc_name
    if remove_nan_skill_ids:
        data = data[~data[kc_name].isnull()]
    else:
        data.ix[data[kc_name].isnull(), kc_name] = 'NaN'

    # Turn skills into single names. Take the first lexicographically if there's more than
    # one, though this can be modified. Only do for non nan skills.
    data[kc_name] = data[kc_name].apply(lambda x: sorted(x.split('~~'))[0])

    LOGGER.info("Total of {:3,d} rows remain after removing NaN skills".format(len(data)))

    # sort by user, time, item, and concept id (if available)
    sort_keys = [USER_ID_KEY, TIME_IDX_KEY, item_id_col]

    data = data.sort_values(sort_keys)
    if drop_duplicates:
        data = data.drop_duplicates(sort_keys)

    # filter for students with >= min_history_length interactions;
    # must be done after removing nan skillz
    data = data.groupby(USER_ID_KEY).filter(lambda x: len(x) >= min_interactions_per_user)
    LOGGER.info("Removed students with <{} interactions ({:3,d} rows remaining)".format(
        min_interactions_per_user, len(data)))

    # limit to first `max_interactions_per_user`
    if max_interactions_per_user is not None:
        old_data_len = len(data)
        data = data.groupby([USER_ID_KEY]).head(max_interactions_per_user)
        LOGGER.info("Filtered for {} max interactions per student ({:3,d} rows removed)".format(
            max_interactions_per_user, old_data_len - len(data)))
        
    with open('questions_item.txt','r') as f:
        question_set = f.readlines()
    question_dict = {}
    for question in question_set:
        delete_line_end_symbol = question.replace('\n','')
        order_question = delete_line_end_symbol.split('*@*')
        question_dict[order_question[1]] = int(order_question[0])
            
    user_ids, data[USER_IDX_KEY] = np.unique(data[USER_ID_KEY], return_inverse=True)
    item_ids, _ = np.unique(data[item_id_col], return_inverse=True)
    data[ITEM_IDX_KEY] = -1
    with open('predict_result/student_question_history.txt','w') as f:
        for i in range(data_length):
            try:
                data[ITEM_IDX_KEY].loc[i] = question_dict[data[item_id_col].loc[i]]
            except KeyError as e:
                continue
            f.write(str(data[ITEM_IDX_KEY].loc[i])+ ',')
    user_ids = user_ids.astype(str)
    item_ids = item_ids.astype(str)
 
    # TODO (yan): refactor the below to avoid code duplication across data sets
    cols_to_keep = [USER_IDX_KEY, ITEM_IDX_KEY, CORRECT_KEY, TIME_IDX_KEY]

    LOGGER.info("Processed data: {:3,d} interactions, {:3,d} students; {:3,d} items "
                .format(len(data), len(user_ids), len(item_ids)))

    return data[cols_to_keep], user_ids, item_ids
