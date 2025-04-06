import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    :param messages_filepath:
    :param categories_filepath:
    :return:
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories)
    return df


def clean_data(df):
    """
    :param df:
    :return:
    """
    ids_col = df['id'].copy(deep=True)
    categories = df['categories'].str.split(pat=";", expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0].values
    category_colnames = [value[:-2] for value in row]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)
    categories['id'] = ids_col
    df.drop(columns=['categories'], inplace=True)
    df = df.merge(categories, on=['id'])
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    :param df:
    :param database_filename:
    :return:
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('CleanData', engine, index=False)
    return


def main():
    """
    :return:
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
