import pandas as pd
import os
from datetime import datetime, timedelta

start_time = datetime.strptime("2025-04-12 00:00", "%Y-%m-%d %H:%M")

directory = '.'


for filename in os.listdir(directory):
    if filename.startswith("5_Dec_") and filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)

        df = pd.read_csv(filepath)

        df['Timestamp'] = [start_time + timedelta(minutes=3 * i) for i in range(len(df))]

        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        
        hourly_df = df.resample('h').mean()
        
        hourly_df.reset_index(inplace=True)

        cols = hourly_df.columns.tolist()
        cols = ['Timestamp'] + [col for col in cols if col != 'Timestamp']
        hourly_df = hourly_df[cols]

        output_filename = f"hourly_{filename}"
        hourly_df.to_csv(output_filename, index=False)
        print(f"Processed and saved: {output_filename}")
