# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """load data given by user

    Args:
    messages_filepath: string. path to the messages file.
    categories_filepath: string. path to the categories file.

    Returns:
    merged pandas dataframe of messages + categories.
    """

    # read data files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    categories = categories.set_index(categories['id'])

    # transform categories dataframe to 36 columns
    categories = categories.categories.str.split(';', expand=True)
    row = categories.iloc[0]

    # rename columns
    category_colnames = [col[:-2] for col in row]
    categories.columns = category_colnames

    # set value of each cell
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """clean dataframe given by user

    Args:
    df: dataframe.

    Returns:
    cleaned pandas dataframe.
    """

    # shows the number before and after cleaning
    print(f"""number of duplicates before cleaning:
        > {df[df.duplicated(keep=False)].count()[0]}""")
    df.drop_duplicates(keep=False, inplace=True)
    print(f"""number of duplicates after cleaning:
        > {df[df.duplicated(keep=False)].count()[0]}""")
    return df


def save_data(df, database_filename):
    """save dataframe

    Args:
    df: dataframe.
    database_filename: string. location where database will be saved

    Returns:
    no return.
    """

    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(database_filename.split('/')[-1], engine, index=False)


def main():
    """main function
    runs the main code, take user inputs in command line
    and do the load/clean/save.

    Args:
    no args.

    Returns:
    no return.
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
