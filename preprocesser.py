import pandas as pd
import re
def preprocess(data):
    # Regular expression to match date-time patterns in WhatsApp messages
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}'

    # Split messages and extract dates
    messages = re.split(pattern, data)[1:]  # Messages after date-time
    dates = re.findall(pattern, data)  # Extract date-time

    # Check for empty data
    if not messages or not dates:
        raise ValueError("No valid date or message data found.")

    # Create DataFrame
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Convert 'message_date' to datetime
    try:
        df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %H:%M')
    except ValueError:
        # Handle alternative date-time formats if needed
        df['message_date'] = pd.to_datetime(df['message_date'], errors='coerce')

    # Rename column
    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []

    # Process each message
    for message in df['user_message']:
        # Split user and message using ':' separator
        entry_split = re.split(r'(?<=\w):\s', message)
        if len(entry_split) > 1:
            users.append(entry_split[0])  # User
            messages.append(entry_split[1])  # Message
        else:
            users.append('group_notification')
            messages.append(entry_split[0])  # System/group message

    # Add parsed data to DataFrame
    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # Extract additional date components
    df['year'] = df['date'].dt.year
    df['only_date'] = df['date'].dt.date
    df['day_name'] = df['date'].dt.day_name()
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hours'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Generate time periods (e.g., "23-00", "00-01")
    df['period'] = df['hours'].apply(lambda hour: f"{hour:02d}-{(hour + 1) % 24:02d}")

    return df
