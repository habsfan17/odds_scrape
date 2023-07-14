import requests
import random
from bs4 import BeautifulSoup
import ast
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import re


from sqlalchemy import create_engine, Column, Integer,Numeric, String,func, and_, text, desc
from sqlalchemy.orm import sessionmaker,declarative_base
from pybettor import convert_odds

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from datetime import date
from datetime import datetime

Base = declarative_base()

class Match(Base):
    __tablename__ = 'Match'
    ID = Column(Integer, primary_key=True, autoincrement=True)
    GameNum = Column(Integer, nullable=False)
    Team = Column(String, nullable=False)
    Score = Column(Integer)
    Sport = Column(Integer, nullable=False)
    Moneyline = Column(Numeric)
    O_U_Number = Column(String)
    O_U_Odds = Column(Numeric)
    Spread_Line = Column(String)
    Spread_Odds = Column(Numeric)
    Date = Column(String)
    Side = Column(String)
    State = Column(String)

class Bet(Base):
    __tablename__ = 'Bet'
    ID = Column(Integer, primary_key=True, autoincrement=True)
    Sport = Column(Integer, nullable=True)
    GameNum = Column(Integer, nullable=True)
    Amount = Column(Numeric, nullable=False)
    Odds = Column(Numeric, nullable=False)
    State = Column(String, nullable=False)
    Description = Column(String, nullable=True)
    Date_Time = Column(String, nullable=False)
    Event_date = Column(String, nullable=False)
    Bookmaker = Column(Integer, nullable=True)


def see_Bets():
    engine = create_engine('sqlite:///odds_scrapper_db.db')
    Session = sessionmaker(bind=engine)
    session = Session()


    # Fetch all bets from the database
    display_option = st.radio("Display Option", ("Pending Bets", "All Bets"))

    if display_option == "Pending Bets":
        query = text("SELECT * FROM Bet WHERE State = 'Pending'")
        all_bets = False
    else:
        query = text("SELECT * FROM Bet")
        all_bets =True

    bets = session.execute(query)
    bets_df = pd.DataFrame(bets, columns=bets.keys())
    
    bets_df['Profit'] = bets_df.apply(lambda row: (row['Amount'] * row['Odds']) - row['Amount'] if row['State'] == 'W'
                                 else 0 if row['State'] == 'Pending'
                                 else 0 if row['State'] == 'Push'
                                 else -row['Amount'], axis=1)
        # Convert 'Event_date' column to datetime format if it is not already in that format
    bets_df['Event_date'] = pd.to_datetime(bets_df['Event_date'])

    # Group by 'Event_date' and calculate the sum of 'Profit' for each day
    profit_per_day = bets_df.groupby('Event_date')['Profit'].sum().reset_index()

    # Sort the dataframe by 'Event_date' in ascending order
    profit_per_day = profit_per_day.sort_values('Event_date')
    avg_profit_per_day = round(profit_per_day['Profit'].mean(),2)
    total_betted = bets_df['Amount'].sum()
    avg_odds = round(bets_df['Odds'].mean(),2)
    odds_ratio = round(1/avg_odds,2)
    betted_per_day = bets_df.groupby('Event_date')['Amount'].sum().reset_index()
    
    avg_bet_per_day = round(betted_per_day['Amount'].mean(),2)
    win_ratio = round(len(bets_df[bets_df['State'] == 'W']) / len(bets_df),2)
    



    



    # Display the dataframe as an editable table
    if all_bets == False:
        edited_data  = st.data_editor(bets_df, num_rows="dynamic")
    else:
        edited_data  = st.data_editor(bets_df)

    # Save changes to the database
    if st.button("Save Changes"):
        for index, row in edited_data.iterrows():
            bet_id = row["ID"]
            sport = row["Sport"]
            gameNum = row["GameNum"]
            amount = row["Amount"]
            odds = row["Odds"]
            state = row["State"]
            description = row["Description"]
            date_time = row["Date_Time"]
            event_date = datetime.strftime(row["Event_date"], "%Y-%m-%d")
            bookmaker = row["Bookmaker"]
            

            query = text("UPDATE Bet SET Sport=:sport, GameNum=:gameNum, Amount=:amount, Odds=:odds, State=:state, "
                         "Description=:description, Date_Time=:date_time, Event_date=:event_date, Bookmaker=:bookmaker "
                         "WHERE ID=:bet_id")
            session.execute(query, {"sport": sport, "gameNum": gameNum, "amount": amount, "odds": odds, "state": state,
                                    "description": description, "date_time": date_time, "event_date": event_date,
                                    "bet_id": bet_id,"bookmaker":bookmaker})
        session.commit()
        st.success("Changes saved successfully.")
    # Create a line plot
    if st.button('See graph'):
        profit_per_day['Cumulative_Profit'] = profit_per_day['Profit'].cumsum()

        new_row = pd.DataFrame({'Event_date': [profit_per_day['Event_date'].min() - pd.DateOffset(days=1)],
                        'Profit': [0],
                        'Cumulative_Profit': [0]})
        
        profit_per_day = pd.concat([new_row, profit_per_day], ignore_index=True)

        # Sort the DataFrame by 'Event_date' in ascending order
        profit_per_day = profit_per_day.sort_values('Event_date')



        # Create the line chart with adjusted profit data
        chart = st.line_chart(profit_per_day.set_index('Event_date')['Cumulative_Profit'])

    st.write('Total betted - $' + str(total_betted))
    st.write('Avg Betted/Day - $' + str(avg_bet_per_day))
    st.write('Avg Profit/Day - $' + str(avg_profit_per_day))
    st.write('Avg Odds - ' + str(avg_odds) + '(' +str(odds_ratio*100) + '%)')
    st.write('Win ratio - ' + str(win_ratio*100)+ '%')





    session.close()




def scrapeMLBodds(start_date, end_date):
    # Set up Selenium options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode
    chrome_options.add_argument("--disable-gpu")

    # Create a Selenium webdriver
    driver = webdriver.Chrome(options=chrome_options)

    engine = create_engine('sqlite:///odds_scrapper_db.db')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    url = "https://www.scoresandodds.com/mlb?date="
    game_number = session.query(func.max(Match.GameNum)).scalar()
    if game_number is None:
        game_number = 0
    else:
        game_number += 1

    date_range = pd.date_range(start_date, end_date).to_pydatetime().tolist()
    game_number_l = []
    data_side_l = []
    team_l = []
    score_l = []
    live_moneyline_l = []
    live_total_runs_l = []
    live_total_odds_l = []
    date_l = []
    state_l = []
    spread_value_l = []
    spread_odds_l = []

    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        full_url = url + date_str

        # Load the URL using Selenium
        driver.get(full_url)


        # Wait for the page to load before scraping
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'event-card')))
        time.sleep(1)  # Pause for 1 second before the next scrape
        # Get the HTML content of the page
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')

        container_cards = soup.find('div', class_='container-body cards')

        # Find all game containers
        game_containers = container_cards.find_all('div', class_='event-card')

        # Loop through each game container
        for game_container in game_containers:
            try:
                state = game_container.thead.find("th").find("span", {"data-field": "state"}).text.strip()
                rows = game_container.tbody.find_all("tr",class_="event-card-row")

                game_number += 1
                # Extract team names
                for row in rows:
                    data_side = row.get("data-side")
                    team = row.find("span", class_="team-name").a.text.strip()
            
                    try:
                        score = row.find("td", class_="event-card-score").text.strip()
                    except:
                        score = 0
                    try:
                        live_moneyline = row.find("td", {"data-field": "live-moneyline"}).find("span", class_="data-value").text.strip()
                    except:
                        live_moneyline = row.find("td", {"data-field": "current-moneyline"}).find("span", class_="data-value").text.strip()
        
                    live_total = row.find("td", {"data-field": "live-total"})

                    live_spread = row.find("td", {"data-field": "live-spread"})

                    if live_spread and live_spread.text.strip() != None:
                       pass
                    else:
                        live_spread = row.find("td", {"data-field": "current-spread"})
     
                    spread_value = float(live_spread.find("span", class_="data-value").text.strip())
                    spread_odds = live_spread.find("small", class_="data-odds").text.strip()
                    if spread_odds == 'even':
                            spread_odds = 100
                    spread_odds = convert_odds(int(spread_odds), cat_in="us", cat_out="dec")[0]

                    if live_total and live_total.text.strip() != None:
                        pass
                    else:
                        live_total = row.find("td", {"data-field": "current-total"})

                    if live_total and live_total != None and live_total.find("span", class_="data-value").text.strip():
                        live_total_runs = live_total.find("span", class_="data-value").text.strip()
                        live_total_odds = (live_total.find("small", class_="data-odds").text.strip())
                        if live_moneyline == 'even':
                            live_moneyline = 100
                        
                        live_moneyline = convert_odds(int(live_moneyline), cat_in="us", cat_out="dec")[0]
                        

                        if live_total_odds == 'even':
                            live_total_odds = 100
                        live_total_odds = convert_odds(int(live_total_odds), cat_in="us", cat_out="dec")[0]
                    else:
                        live_total_runs = None
                        live_total_odds = None



                    game_number_l.append(game_number)
                    data_side_l.append(data_side)
                    team_l.append(team)
                    score_l.append(score)
                    live_moneyline_l.append(live_moneyline)
                    live_total_runs_l.append(live_total_runs)
                    live_total_odds_l.append(live_total_odds)
                    spread_value_l.append(spread_value)
                    spread_odds_l.append(spread_odds)
                    date_l.append(date_str)
                    state_l.append(state)

                    # Display the scraped information on the Streamlit page
                    st.write(f"Game Number: {game_number}")
                    st.write(f"Side: {data_side}")
                    st.write(f"Team: {team}")
                    st.write(f"Score: {score}")
                    st.write(f"Live Moneyline: {live_moneyline}")
                    st.write(f"Live Total Runs: {live_total_runs}")
                    st.write(f"Live Total Odds: {live_total_odds}")
                    st.write(f"Live Spread Runs: {spread_value}")
                    st.write(f"Live Spread Odds: {spread_odds}")
                    st.write(f"State: {state}")
                    st.write("_")  # Add a separator between games



                st.write("---")  # Add a separator between games
            except:
                pass

    if st.button("OK"):
        session.query(Match).filter(and_(Match.Sport == 1, Match.Date.between(start_date, end_date))).delete()
        for i in range(len(game_number_l)):
            match = Match(
                GameNum=game_number_l[i],
                Team=team_l[i],
                Score=score_l[i],
                Sport=1,  # Update with the correct sport value
                Moneyline=live_moneyline_l[i],
                O_U_Number=live_total_runs_l[i],
                O_U_Odds=live_total_odds_l[i],
                Spread_Line=spread_value_l[i],
                Spread_Odds=spread_odds_l[i],
                Side=data_side_l[i],
                Date=date_l[i],
                State=state_l[i]
            )
            session.add(match)

        session.commit()
        st.write(f"Rows added to the database for date {date_str} successfully.")

    session.close()
    driver.quit()  # Quit the Selenium webdriver
    st.write("Scraping and database update complete.")

def analyzeMLBodds(start_date, end_date):
    engine = create_engine('sqlite:///odds_scrapper_db.db')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    query = text("SELECT * FROM Match WHERE Sport = 1 AND Date BETWEEN :start_date AND :end_date;")
    result = session.execute(query, {"start_date": start_date, "end_date": end_date})

    bet_amount = 5

    df = pd.DataFrame(result.fetchall(), columns=result.keys())
    df = df[df['State'] == 'FINAL']
    df['Winner'] = df.groupby('GameNum')['Score'].transform(max) == df['Score']
    df['O_U_Winner'] = df.apply(lambda row: 'Over' if (float(row['Score']) > float(row['O_U_Number'][1:])) and (row['O_U_Number'][0] == 'o') else 'Under', axis=1)
    df['Fav_dog'] = df.apply(lambda row: 'Even' if row['Moneyline'] == min(df[df['GameNum'] == row['GameNum']]['Moneyline']) and row['Moneyline'] == max(df[df['GameNum'] == row['GameNum']]['Moneyline']) else ('Favorite' if row['Moneyline'] == min(df[df['GameNum'] == row['GameNum']]['Moneyline']) else 'Underdog'), axis=1)
    df['Opp_Score'] = df.apply(lambda row: df[(df['GameNum'] == row['GameNum']) & (df['Side'] != row['Side'])]['Score'].iloc[0] if not df[(df['GameNum'] == row['GameNum']) & (df['Side'] != row['Side'])].empty else None, axis=1)
    df['Opp'] = df.apply(lambda row: df[(df['GameNum'] == row['GameNum']) & (df['Side'] != row['Side'])]['Team'].iloc[0] if not df[(df['GameNum'] == row['GameNum']) & (df['Side'] != row['Side'])].empty else None, axis=1)
    game_scores = df.groupby(['GameNum', 'Team'])['Score'].sum().reset_index()

    merged_df = df.merge(game_scores, on=['GameNum', 'Team'])
    merged_df['Runs_Allowed'] = merged_df.groupby('GameNum')['Score_y'].transform('sum') - merged_df['Score_y']

    # Sort the dataframe by date in descending order
    df_sorted = df.sort_values('Date', ascending=False)

    # Convert start_date and end_date to Timestamp objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the DataFrame by start_date and end_date
    df_sorted['Date'] = pd.to_datetime(df_sorted['Date'])
    df_sorted = df_sorted[(df_sorted['Date'] > start_date) & (df_sorted['Date'] < end_date)]

    # Create new columns to store the Pythagorean Winning Percentage for both the team and the opponent
    df_sorted['Pyth_W_%'] = np.nan
    df_sorted['Opp_Pyth_W_%'] = np.nan
    df_sorted['gap'] = np.nan
    df_sorted['OOdds'] = np.nan
    df_sorted['OOdds_gap'] = np.nan

    # Iterate over each row
    for team in df_sorted['Team'].unique():
        # Get all the rows for the current team
        team_rows = df_sorted[df_sorted['Team'] == team].sort_values('Date', ascending=False)

        # Iterate over each row of the current team
        for index, row in team_rows.iterrows():
            opponent = row['Opp']
            game_date = pd.to_datetime(row['Date'])  # Convert the game date to Timestamp

            # Get the last 25 games of the corresponding team up to the current game date
            last_x_games = team_rows[team_rows['Date'] <= game_date].head(50)
            opp_last_x_games = df_sorted[(df_sorted['Team'] == opponent) & (df_sorted['Date'] <= game_date)].head(50)

            # Check if there are at least 25 games available for the team
            if len(last_x_games) >= 50:
                # Calculate the runs scored and runs allowed for the last 25 games
                runs_scored = last_x_games['Score'].sum()
                runs_allowed = last_x_games['Opp_Score'].sum()
                pythagorean_win_percentage = (runs_scored ** 1.87) / ((runs_scored ** 1.87) + (runs_allowed ** 1.87))

                opp_runs_scored = opp_last_x_games['Score'].sum()
                opp_runs_allowed = opp_last_x_games['Opp_Score'].sum()
                opp_pythagorean_win_percentage = (opp_runs_scored ** 2) / ((opp_runs_scored ** 2) + (opp_runs_allowed ** 2))
                gap = pythagorean_win_percentage - opp_pythagorean_win_percentage

                if row['Side'] == 'home':
                    odjunkOdds = (1 / (0.5 * (1 + gap + 0.04)))
                else:
                    odjunkOdds = (1 / (0.5 * (1 + gap - 0.04)))

                odjunkOdds_gap = (row['Moneyline'] - odjunkOdds)/row['Moneyline']

                # Assign the calculated Pythagorean Winning Percentage and gap to the respective columns
                df_sorted.at[index, 'Pyth_W_%'] = round(pythagorean_win_percentage,2)
                df_sorted.at[index, 'Opp_Pyth_W_%'] = round(opp_pythagorean_win_percentage,2)
                df_sorted.at[index, 'gap'] = round(gap,2)
                df_sorted.at[index, 'OOdds'] = round(odjunkOdds,2)
                df_sorted.at[index, 'OOdds_gap'] = round(odjunkOdds_gap,2)


        # Create a new dataframe to store the step information
    steps = [0.5, 0.4, 0.3, 0.25,0.2 ,0.15,0.125, 0.1,0.075, 0.05,0.025, 0, -0.1, -0.2, -0.3, -0.4, -0.5]
    step_data = []

    for i in range(len(steps) - 1):
        step_start = steps[i]
        step_end = steps[i + 1]


        step_df = df_sorted[(df_sorted['OOdds_gap'] < step_start) & (df_sorted['OOdds_gap'] >= step_end)]
        step_count = len(step_df)

        step_df['step_profit'] = step_df.apply(lambda row: (row['Moneyline'] * bet_amount) - bet_amount if row['Winner'] else -bet_amount, axis=1)
        step_profit_sum = step_df['step_profit'].sum()

        step_data.append({
            'Step Start': step_start,
            'Step End': step_end,
            'Bet Count': step_count,
            'Bet Profit':step_profit_sum,
            'Over Step Profit %': round((step_profit_sum / (step_count * bet_amount))*100,2)

        })

    step_counts_df = pd.DataFrame(step_data)
    df_sorted = df_sorted[['Date','Team','Score','Opp','Opp_Score','Moneyline',
                           'OOdds','OOdds_gap']]
    pd.options.display.width = 0
    pd.set_option('display.max_columns', None)
    st.dataframe(df_sorted)
    st.dataframe(step_counts_df)

    session.close()

def mlbDailyGames(date):
    engine = create_engine('sqlite:///odds_scrapper_db.db')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
     # Get the distinct teams from the query
    display_option = st.radio("Display Option", ("Ongoing", "All"))

    if display_option == "Ongoing":
        query = text("SELECT * FROM Match WHERE Sport = 1 AND State != 'FINAL' AND Date = :game_date;")
    else:
        query = text("SELECT * FROM Match WHERE Sport = 1  AND Date = :game_date;")

    result = session.execute(query, {'game_date': date})
    result_df = pd.DataFrame(result, columns=result.keys())
    matches_df = result_df
    teams= set(result_df['Team'].to_list())
    bookmakers = session.execute(text('SELECT * FROM Bookmaker'))
    bookmakers_df = pd.DataFrame(bookmakers, columns=bookmakers.keys())
    


    # Get the last 50 games for each team
    all_games = []
    for team in teams:
        query = text("SELECT * FROM Match WHERE Team = :team AND Sport = 1 AND State == 'FINAL' AND Date <= :game_date ORDER BY Date DESC LIMIT 75;")
        result = session.execute(query, {'team': team, 'game_date': date})
        games = result.fetchall()
        all_games.extend(games)


    # Create DataFrame from the results
    df = pd.DataFrame(all_games, columns=result.keys())
    team_stats = []
    for team in teams:
        team_df = df[df['Team'] == team]
        total_runs_scored = team_df['Score'].sum()
        total_runs_allowed = df[df['GameNum'].isin(team_df['GameNum'])]['Score'].sum() - total_runs_scored
        team_stats.append((team, total_runs_scored, total_runs_allowed))


    team_stats_df = pd.DataFrame(team_stats, columns=['Team', 'Total Runs Scored', 'Total Runs Allowed'])

    team_stats_df['Pyth_W_%'] = (team_stats_df['Total Runs Scored'] ** 1.87) / ((team_stats_df['Total Runs Scored'] ** 1.87) + (team_stats_df['Total Runs Allowed'] ** 1.87))
    team_stats_df = team_stats_df.sort_values('Pyth_W_%', ascending=False)

    result_df = result_df.merge(team_stats_df[['Team', 'Pyth_W_%']], on='Team', how='left')
    result_df['Opp'] = result_df.apply(lambda row: result_df[(result_df['GameNum'] == row['GameNum']) & (result_df['Side'] != row['Side'])]['Team'].iloc[0] if len(result_df[(result_df['GameNum'] == row['GameNum']) & (result_df['Side'] != row['Side'])]) > 0 else '', axis=1)
    result_df['Opp_Pyth_W_%'] = result_df.apply(lambda row: result_df[(result_df['GameNum'] == row['GameNum']) & (result_df['Side'] != row['Side'])]['Pyth_W_%'].iloc[0] if len(result_df[(result_df['GameNum'] == row['GameNum']) & (result_df['Side'] != row['Side'])]) > 0 else None, axis=1)
    result_df['gap'] = result_df['Pyth_W_%'] - result_df['Opp_Pyth_W_%']
    result_df['OdjunkOdds'] = result_df.apply(lambda row: round((1 / (0.5 * (1 + row['gap'] + 0.04))), 2) if row['Side'] == 'home' else round((1 / (0.5 * (1 + row['gap'] - 0.04))), 2), axis=1)
    result_df['Moneyline'] = result_df['Moneyline'].apply(lambda x: x if x != '' else 99)
    result_df['Moneyline'] = result_df['Moneyline'].astype(float)
    result_df['Edge%'] = ((result_df['Moneyline'] - result_df['OdjunkOdds']) / result_df['OdjunkOdds']) * 100
    result_df['Edge%'] = result_df['Edge%'].round(1)
    result_df['Spread_Line'] = result_df['Spread_Line'].apply(lambda x: x if x != '' else 99)
    result_df['Spread_Line'] = result_df['Spread_Line'].astype(float)
    result_df['Spread_Odds'] = result_df['Spread_Odds'].apply(lambda x: x if x != '' else 99)
    result_df['Spread_Odds'] = result_df['Spread_Odds'].astype(float)
    result_df['Spread'] = result_df['Spread_Odds'].astype(str) + '(' + result_df['Spread_Line'].astype(str) + ')'
    result_df['Side'] = result_df['Side'].replace({'away': 'a', 'home': 'h'})
    result_df = result_df[['Date', 'GameNum', 'Side', 'Team', 'Opp', 'Moneyline', 'OdjunkOdds', 'Edge%', 'State', 'Spread']]

    # Find the highest Edge% for each GameNum group and sort the groups
    positive_edge_teams = result_df[result_df['Edge%'] > 2.5]['Team'].unique()

    # Divide the screen into two columns

    # Render game tables in the left column

    st.title('Games')
    sorted_game_nums = result_df.groupby('GameNum')['Edge%'].max().sort_values(ascending=False).index

    for game_num in sorted_game_nums:
        game_data = result_df[result_df['GameNum'] == game_num].sort_values('Edge%', ascending=False)    
        teams = game_data['Team'].to_list()
        st.write(f"GameNum: {game_num}")

        game_data = game_data[['Side', 'Team', 'Moneyline','Spread','OdjunkOdds', 'Edge%', 'State']]
        game_data = game_data.rename(columns={'Add Bet': '', 'Side': 'S', 'Moneyline': 'ML', 'OdjunkOdds': 'OOdds'})
        game_data['Edge%'] = game_data['Edge%'].round(1).astype(str)
        game_data['ML'] = game_data['ML'].round(2).astype(str)
        game_data['OOdds'] = game_data['OOdds'].round(2).astype(str)

        col1, col2 = st.columns([3, 1])

        with col1:
            st.dataframe(game_data)

        with col2:
   
            with st.form(key=f"bet_form_{game_num}"):
                st.write(f"Betting Form #{game_num}")
                bookmaker = st.selectbox("Select Bookmaker", bookmakers_df['Nom'].to_list())
                team = st.selectbox("Select Side", teams)  # Add input for selecting the side
                odds = st.number_input("Odds", value=float(game_data['ML'].to_list()[0]))  # Add input for entering the odds
                amount = st.number_input("Amount Bet", value=10)  # Add input for entering the amount bet
                submit_button = st.form_submit_button(label="Place Bet")  # Add a submit button

                if submit_button:
                    bookmaker_id = int(bookmakers_df.loc[bookmakers_df['Nom'] == bookmaker, 'ID'].values[0])
                  
                    gameNum = matches_df[(matches_df['GameNum'] == game_num) & (matches_df['Team'] == team)]['ID'].to_list()[0]
                    opponents = ' '.join(matches_df[(matches_df['GameNum'] == game_num) & (matches_df['Team'] != team)]['Team'].to_list())
                    current_time = datetime.now().strftime('%Y-%m-%d|%H:%M')

                    description = str(amount) + '$ bet on ' + team + ' for the match on ' + date +' against ' + opponents

                    new_bet = Bet(
                        Sport=1,
                        GameNum=gameNum, 
                        Amount=amount,
                        Odds=odds,
                        State='Pending',
                        Description = description,
                        Date_Time = current_time,
                        Event_date = date,
                        Bookmaker=bookmaker_id

                    )
                    session.add(new_bet)
                    st.write(f"Added {description}")
                    session.commit()

    session.close()


def scrapemmaFightMatrixLinks():
    # Set up Selenium options



    # engine = create_engine('sqlite:///odds_scrapper_db.db')
    # Base.metadata.create_all(engine)
    # Session = sessionmaker(bind=engine)
    # session = Session()
    url = "https://www.fightmatrix.com/mma-ranks/"
 
   

    weight_l = ['heavyweight-265-lbs','light-heavyweight-185-205-lbs','middleweight','welterweight','lightweight','featherweight','bantamweight','flyweight',]

    selected_weight = 'flyweight'
    fighter_id_l = []
    fighter_name_l = []
    for n in range(1,35):
        full_url = url + selected_weight + '/?PageNum=' + str(n)

        # Load the URL using Selenium
        html_content = requests.get(full_url)
        soup = BeautifulSoup(html_content.text, 'html.parser')
        table = soup.find('table', class_='tblRank')

        a_links = table.find_all('a', class_='sherLink')



        for a in a_links:
            
            href = a.get('href')
            last_slash_index = href.rfind("/")
            second_last_slash_index = href.rfind("/", 0, last_slash_index)
            fighter_id = href[second_last_slash_index+1:last_slash_index]
            fighter_name = a.get('name')
            
            fighter_id_l.append(fighter_id)
            fighter_name_l.append(fighter_name)

    df = pd.DataFrame({'fighter_id': fighter_id_l, 'fighter_name': fighter_name_l})
    # df.to_csv('flyweight.csv', index=False)

def scrapeFighterSherdogTapologyLinks():
    df = pd.read_csv('Fighters_ids_csv.csv', delimiter=';')

    url = 'https://www.fightmatrix.com/fighter-profile/J/'
    fight_matrix_id_l = []
    tapology_id_l = []

    for index, row in df.iterrows():
        fight_matrix_id = row['FightMatrixID']
        full_url = url + str(fight_matrix_id)
        
        try:
            session = requests.Session()
            retry_strategy = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
            response = session.get(full_url, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error occurred for FightMatrix ID {fight_matrix_id}: {str(e)}")
            continue
        
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', class_='tblRank')
        td_link = table.find('td', class_='tdRankHead')
        tapology = td_link.find('p')
        tapology_id = tapology.find('em').a.get("href").split('/')[-1]
        
        fight_matrix_id_l.append(fight_matrix_id)
        tapology_id_l.append(tapology_id)
        time.sleep(random.uniform(1, 3))  # Random delay between 1 and 3 seconds

    data = {'fight_matrix_id': fight_matrix_id_l, 'tapology_id': tapology_id_l}
    df = pd.DataFrame(data)
    df.to_csv('fighters_ids_ref_tap.csv', index=False)
    print(index)

def scrapeMMArankings():
    df = pd.read_csv('Fighters_ids_csv.csv',delimiter=';')

    weight = 'flyweight'
    session = requests.Session()
    retry_strategy = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    url_FM = 'https://www.fightmatrix.com/fighter-profile/J/'
    # url_S = 'https://www.sherdog.com/fighter/J-J-'
    fight_matrix_id_l = []
    date_rank_l = []
    points_l = []

    for index, row in df.iterrows():
        if index < 5:

            fight_matrix_id = row['FightMatrixID']
            if row['Weight'] == weight:
                full_url_FM = url_FM + str(fight_matrix_id) +'/'
                html_content_FM = session.get(full_url_FM)
                soup_FM = BeautifulSoup(html_content_FM.text, 'html.parser')
                try:
                    tbl = soup_FM.find_all('table', class_='tblRank')[1]
                    trs = tbl.find_all('tr')
                    for tr in trs:
                        try:
                            date_rank = tr.find_all('td', class_='tdRank')
                        except:
                            date_rank = tr.find_all('td', class_='tdRankAlt')
                        if date_rank:
                            points = date_rank[-1].find('table').find('td', class_='tdBar').text.strip()
                            date_rank = date_rank[0].a.text.strip()
                            fight_matrix_id_l.append(fight_matrix_id)
                            date_rank_l.append(date_rank)
                            points_l.append(points)
                            print(fight_matrix_id)
                except:
                    pass

    df = pd.DataFrame({'FightMatrixID': fight_matrix_id_l,
                    'DateRank': date_rank_l,
                    'Points': points_l})

    # Save DataFrame to CSV
    df.to_csv('flyweight.csv', index=False)

def scrapeMMAfights():
    df = pd.read_csv('Fighters_ids_csv.csv',delimiter=';')

    weight = 'heavyweight-265-lbs'
    chrome_options = Options()
    chrome_options.add_argument("--ignore-certificate-errors")

    # Create a Selenium webdriver
    driver = webdriver.Chrome(options=chrome_options)

    # url_FM = 'https://www.fightmatrix.com/fighter-profile/J/'
    url = 'https://www.sherdog.com/fighter/J-J-'
    sherdog_id_l = []
    date_l = []
    result_l = []
    method_l = []
    round_l = []
    opponent_id_l = []
    

    for index, row in df.iterrows():
        if index < 5:

            sherdog_id = row['Sherdog ID']
            if row['Weight'] == weight:
                
                full_url = url + str(sherdog_id) +'/'
                driver.get(full_url)
                html_content = driver.page_source
                soup= BeautifulSoup(html_content.text, 'html.parser')
                print(soup)
                table = soup.find('table', class_='new_table fighter').tbody
                trs = table.find_all('tr:not(.table_head)')
                for tr in trs:
                    tds = tr.find_all('td')
                    for td in tds:
                        print(td.text.strip())
                driver.quit()




            
          



    # # Save DataFrame to CSV
    # df.to_csv('flyweight.csv', index=False)

                













def app():
    today = date.today().strftime("%Y-%m-%d")
    st.title("Betting assistant")
    leagues = ["MLB"]

    selected_league = st.sidebar.selectbox("Select a league", leagues)
    


    if selected_league:
        option = st.sidebar.radio("Choose an option", ["Scrape", "Daily Games", "Analysis", "My Bets","MMA"])


    if option == "Scrape":

        st.write("Scrape")

        start_date = st.text_input("Enter the start Date: ")
        end_date = st.text_input("Enter the end Date: ")


        if start_date and end_date:
            scrapeMLBodds(start_date, end_date)
    elif option == "Analysis":
        st.write("Analysis")

        start_date = st.text_input("Enter the start Date: ")
        end_date = st.text_input("Enter the end Date: ")
        if start_date and end_date:
            analyzeMLBodds(start_date, end_date)
    elif option == "Daily Games":
        st.write("Daily Games")

        date_inputed = st.text_input("Enter the Date:", today)

        if date:
            mlbDailyGames(date_inputed)
    elif option == "My Bets":
        st.write("My bets:")
        see_Bets()
    elif option == 'MMA':
        scrapeFighterSherdogTapologyLinks()



        

def simulate_betting(num_bets, win_prob, bet_amount, bankroll):
    
    for _ in range(num_bets):
        if random.random() < win_prob:
            bankroll += bet_amount * (1/win_prob - 1)
        else:
            bankroll -= bet_amount


    return bankroll

def calculate_chances(num_simulations, num_bets, win_prob, bet_amount, bankroll):
    hit_zero_count = 0
    double_up_count = 0
    final_bankroll_l = []

    for _ in range(num_simulations):
        final_bankroll = simulate_betting(num_bets, win_prob, bet_amount, bankroll)

        final_bankroll_l.append(final_bankroll)
        if final_bankroll > bankroll*2:
            double_up_count += 1
        elif final_bankroll < 0:
            hit_zero_count += 1
        else:
            pass


    hit_zero_chance = hit_zero_count / num_simulations
    double_up_chance = double_up_count / num_simulations

    return hit_zero_chance, double_up_chance, final_bankroll_l




if __name__ == "__main__":
    app()
