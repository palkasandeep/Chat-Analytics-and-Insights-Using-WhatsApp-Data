import pandas as pd
from collections import Counter
from urlextract import URLExtract
from wordcloud import WordCloud
import emoji

# Create the object for URL extract
extract = URLExtract()

def fetch_stats(selected_user, df):
    # Filter DataFrame if a specific user is selected
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    # Total messages
    num_messages = df.shape[0]

    # Total words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # Total media queries
    num_mediaquery = df[df['message'] == '<Media omitted>\n'].shape[0]
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_mediaquery, len(links)

def most_busy(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(columns={
        'message': 'percent',
        'user': 'name'
    })

    return x, df

def create_wordcloud(selected_user, df):
    # Filter DataFrame if a specific user is selected
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    # Initialize WordCloud
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

    # Generate word cloud from messages
    df_wc = wc.generate(df['message'].str.cat(sep=" "))

    return df_wc

def most_common_words(selected_user, df):
    with open('stopwords.txt', 'r') as f:
        stop_words = f.read().splitlines()  # Reads each line and splits by newline

    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    # Use Counter to count occurrences of words
    word_count = Counter(words)

    # Convert to DataFrame
    return_df = pd.DataFrame(word_count.most_common(20), columns=['Word', 'Count'])
    return return_df




def emoji_helper(selected_user, df):
    # Filter the data for a specific user if selected
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    # Initialize an empty list to collect emojis
    emojis = []

    # Extract emojis from each message
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])  # Adjust for your emoji library version

    # Create a DataFrame of emojis with their counts
    emoji_list = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))),
                              columns=['Emoji', 'Count'])

    return emoji_list


def monthly_timeline(selected_user, df):
    """
    Generate a timeline of monthly messages for a given user or overall.

    Parameters:
    - selected_user: User for whom the timeline is to be generated ('overall' for all users).
    - df: Preprocessed DataFrame with chat data.

    Returns:
    - DataFrame containing the timeline with columns 'year', 'month_num', 'month', 'message', and 'time'.
    """
    # Filter for the selected user if not 'overall'
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    # Group by year, month_num, and month, and count the number of messages
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    # Add a new column for time in "Month-Year" format
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)

    return timeline


def daily_timeline(selected_user,df):
    if selected_user!='overall':
        df=df[df['user']==selected_user]
    daily_time=df.groupby('only_date').count()['message'].reset_index()

    return daily_time
def week_activity_map(selected_user,df):
    if selected_user!='overall':
        df=df[df['user']==selected_user]
    return df['day_name'].value_counts()

def monthly_activity(selected_user,df):
     if selected_user!='overall':
         df=df[df['user']==selected_user]
     return df['month'].value_counts()


def heat_map(selected_user, df):
    if selected_user != 'overall':
        # Filter the dataframe for the selected user
        df = df[df['user'] == selected_user]

    # Create the heatmap data by pivoting the table
    heat_mapp = df.pivot_table(
        index='day_name',
        columns='period',
        values='message',
        aggfunc='count'
    ).fillna(0)  # Fill missing values with 0

    return heat_mapp
