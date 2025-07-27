import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests
from collections import Counter
from io import StringIO
from typing import Tuple, List
from scipy.stats import chi2_contingency

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Powerball Number Analyzer",
    page_icon="🎱"
)

# --- Data Loading ---
DATA_URL = "https://data.ny.gov/api/views/d6yy-54nr/rows.csv?accessType=DOWNLOAD"

@st.cache_data(ttl=6 * 60 * 60)
def load_data():
    """
    Loads historical Powerball data from the NY Data.gov source.
    """
    try:
        response = requests.get(DATA_URL)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Read the CSV data from the response text
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        
        # --- Data Cleaning and Preparation ---
        df['Draw Date'] = pd.to_datetime(df['Draw Date'], format='%m/%d/%Y')
        
        # Split the 'Winning Numbers' string into a list of numbers
        # And convert them to integers
        winning_numbers_split = df['Winning Numbers'].str.split(expand=True)
        
        # Extract the main numbers and the powerball
        main_numbers = winning_numbers_split.iloc[:, :-1].astype(int)
        powerball = winning_numbers_split.iloc[:, -1].astype(int)
        
        df['Powerball'] = powerball
        
        # Store main numbers as a list in a new column
        df['Main Numbers'] = main_numbers.values.tolist()
        
        # Keep only the columns we need and set a proper index
        df_cleaned = df[['Draw Date', 'Main Numbers', 'Powerball']].copy()
        df_cleaned.set_index('Draw Date', inplace=True)
        df_cleaned.sort_index(ascending=False, inplace=True)
        
        return df_cleaned

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred while processing the data: {e}")
        return None

# --- Analysis Functions ---
def calculate_frequencies(data: pd.DataFrame) -> Tuple[Counter, Counter]:
    """Calculates the frequency of main numbers and the Powerball number."""
    all_main_numbers = [num for sublist in data['Main Numbers'] for num in sublist]
    main_freq = Counter(all_main_numbers)
    powerball_freq = Counter(data['Powerball'])
    return main_freq, powerball_freq

def get_hot_cold_numbers(freq: Counter, count: int = 5) -> Tuple[List[int], List[int]]:
    """Gets the most (hot) and least (cold) common numbers."""
    # Ensure all numbers from 1-69 are present for a fair comparison
    full_freq = Counter({i: freq.get(i, 0) for i in range(1, 70)})
    hot = [num for num, _ in full_freq.most_common(count)]
    cold = [num for num, _ in full_freq.most_common()[:-count-1:-1]]
    return hot, cold

def analyze_odd_even(data: pd.DataFrame) -> pd.Series:
    """Analyzes the odd/even distribution of main numbers for each draw."""
    return data['Main Numbers'].apply(lambda nums: sum(1 for num in nums if num % 2 != 0)).value_counts()

def analyze_low_high(data: pd.DataFrame) -> pd.Series:
    """Analyzes the low/high distribution of main numbers for each draw."""
    # Low numbers are 1-35, high are 36-69
    return data['Main Numbers'].apply(lambda nums: sum(1 for num in nums if num <= 35)).value_counts()

# --- Number Generation ---
def generate_random_numbers() -> Tuple[List[int], int]:
    """Generates a standard random set of Powerball numbers."""
    main_numbers = sorted(np.random.choice(range(1, 70), 5, replace=False).tolist())
    powerball = np.random.randint(1, 27)
    return main_numbers, powerball

def generate_from_frequency(freq: Counter, mode: str) -> Tuple[List[int], int]:
    """Generates numbers biased towards hot or cold frequencies."""
    population = list(freq.keys())
    weights = np.array(list(freq.values()))
    
    if mode == 'hot':
        probabilities = weights / weights.sum()
    else: # cold
        probabilities = (1 / weights) / (1 / weights).sum()
        
    main_numbers = sorted(np.random.choice(population, 5, replace=False, p=probabilities).tolist())
    powerball = np.random.randint(1, 27) # Powerball is always random
    return main_numbers, powerball

def generate_statistical_profile(data: pd.DataFrame) -> Tuple[List[int], int]:
    """
    Generates numbers that fit a common statistical profile but are less likely
    to be chosen by humans (e.g., avoiding all low numbers for birthdays).
    """
    # Determine a common odd/even split based on history
    odd_even_dist = analyze_odd_even(data)
    num_odd = np.random.choice(odd_even_dist.index, p=odd_even_dist.values / odd_even_dist.sum())
    num_even = 5 - num_odd
    
    # Determine a common low/high split
    low_high_dist = analyze_low_high(data)
    num_low = np.random.choice(low_high_dist.index, p=low_high_dist.values / low_high_dist.sum())
    num_high = 5 - num_low

    # Generate numbers that fit this profile
    odd_numbers = np.arange(1, 70, 2)
    even_numbers = np.arange(2, 70, 2)
    low_numbers = np.arange(1, 36)
    high_numbers = np.arange(36, 70)

    main_numbers = set()
    
    # This is a simplified approach; a more complex one could ensure perfect statistical matching
    # For now, we'll just pull the required number of odd/even and hope the low/high works out
    # A better approach might use constraint satisfaction, but this is good for a start.
    
    selected_odds = np.random.choice(odd_numbers, num_odd, replace=False)
    selected_evens = np.random.choice(even_numbers, num_even, replace=False)
    
    main_numbers.update(selected_odds)
    main_numbers.update(selected_evens)

    while len(main_numbers) < 5: # In case of duplicates from different categories
        main_numbers.add(np.random.randint(1, 70))

    powerball = np.random.randint(1, 27)
    return sorted(list(main_numbers))[:5], powerball


# --- Markov Chain Functions ---
def get_number_bin(num):
    """Assigns a number to a bin."""
    if 1 <= num <= 9: return "1-9"
    if 10 <= num <= 19: return "10-19"
    if 20 <= num <= 29: return "20-29"
    if 30 <= num <= 39: return "30-39"
    if 40 <= num <= 49: return "40-49"
    if 50 <= num <= 59: return "50-59"
    if 60 <= num <= 69: return "60-69"
    return None

def build_markov_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Builds a transition matrix between number bins."""
    bin_transitions = []
    for num_list in data['Main Numbers']:
        bins = [get_number_bin(n) for n in sorted(num_list)]
        bin_transitions.extend(list(zip(bins, bins[1:])))
    
    counts = Counter(bin_transitions)
    bins = ["1-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69"]
    matrix = pd.DataFrame(0, index=bins, columns=bins)
    
    for (start, end), count in counts.items():
        matrix.loc[start, end] = count
        
    row_sums = matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1 # Avoid division by zero
    
    return matrix.div(row_sums, axis=0)

def generate_with_markov(matrix: pd.DataFrame) -> List[int]:
    """Generates numbers by walking the Markov transition matrix."""
    numbers = []
    bins = matrix.index.tolist()
    
    # Start with a random bin
    current_bin = np.random.choice(bins)
    
    while len(numbers) < 5:
        # Get a random number from the current bin
        if current_bin == "1-9": num = np.random.randint(1, 10)
        elif current_bin == "60-69": num = np.random.randint(60, 70)
        else:
            low, high = map(int, current_bin.split('-'))
            num = np.random.randint(low, high + 1)
        
        if num not in numbers:
            numbers.append(num)
            
        # Move to the next bin based on transition probabilities
        current_bin = np.random.choice(bins, p=matrix.loc[current_bin].values)
        
    return sorted(numbers)

def generate_ticket(strategy: str, main_freq: Counter, markov_matrix: pd.DataFrame, data: pd.DataFrame) -> Tuple[List[int], int]:
    """Helper function to dispatch to the correct generator."""
    if strategy == "Purely Random":
        return generate_random_numbers()
    elif strategy == "Hot Numbers":
        return generate_from_frequency(main_freq, mode='hot')
    elif strategy == "Cold Numbers":
        return generate_from_frequency(main_freq, mode='cold')
    elif strategy == "Statistical Profile":
        return generate_statistical_profile(data)
    elif strategy == "Markov":
        main_nums = generate_with_markov(markov_matrix)
        pb = np.random.randint(1, 27)
        return main_nums, pb
    return generate_random_numbers() # Default fallback

def check_ticket(ticket: Tuple[List[int], int], data: pd.DataFrame) -> pd.DataFrame:
    """Checks if a given ticket has won in the past."""
    main_numbers, powerball = ticket
    
    # Ensure main numbers are sorted for a consistent check
    main_numbers = sorted(main_numbers)
    
    # Find draws where the Powerball matches
    powerball_matches = data[data['Powerball'] == powerball]
    
    # From those, find draws where the main numbers also match
    winners = powerball_matches[powerball_matches['Main Numbers'].apply(lambda x: sorted(x) == main_numbers)]
    
    return winners

def calculate_winnings(ticket: Tuple[List[int], int], winning_draw: pd.Series) -> Tuple[str, int]:
    """Calculates the prize tier and amount for a ticket against a winning draw."""
    main_numbers, powerball = ticket
    winning_main = winning_draw['Main Numbers']
    winning_pb = winning_draw['Powerball']

    matched_main = len(set(main_numbers) & set(winning_main))
    matched_pb = (powerball == winning_pb)

    prize_structure = {
        (5, True): ("Jackpot", 1_000_000_000),
        (5, False): ("Match 5", 1_000_000),
        (4, True): ("Match 4 + PB", 50_000),
        (4, False): ("Match 4", 100),
        (3, True): ("Match 3 + PB", 100),
        (3, False): ("Match 3", 7),
        (2, True): ("Match 2 + PB", 7),
        (1, True): ("Match 1 + PB", 4),
        (0, True): ("Match PB", 4),
    }
    
    tier, amount = prize_structure.get((matched_main, matched_pb), ("No Prize", 0))
    return tier, amount

def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("🎱 Powerball Number Analyzer")
    st.markdown("An interactive tool to explore historical Powerball data.")

    # Load the full data
    full_data = load_data()

    if full_data is not None and not full_data.empty:
        st.sidebar.success("Data loaded successfully!")
        
        # --- Sidebar Controls ---
        max_draws = len(full_data)
        num_drawings = st.sidebar.slider(
            "Number of Past Drawings to Analyze", 
            10, max_draws, min(1000, max_draws), 100
        )
        data = full_data.head(num_drawings)
        st.sidebar.metric("Analyzing Recent Draws", f"{len(data):,}")
        
        # Pre-calculate for generator use
        main_freq, powerball_freq = calculate_frequencies(data)
        markov_matrix = build_markov_matrix(data)

        # --- Number Generation Section ---
        st.sidebar.header("Number Generation Suite")
        num_tickets = st.sidebar.slider("How many tickets to generate?", 1, 50, 5)
        gen_strategy = st.sidebar.selectbox(
            "Choose a generation strategy:",
            ("Purely Random", "Hot Numbers", "Cold Numbers", "Statistical Profile", "Markov", "Mixed")
        )
        exclude_past_winners = st.sidebar.checkbox("Exclude Past Winning #s")

        if st.sidebar.button("Generate Numbers"):
            generated_tickets = []
            strategies = ["Purely Random", "Hot Numbers", "Cold Numbers", "Statistical Profile", "Markov"]

            with st.spinner(f"Generating {num_tickets} unique tickets..."):
                if gen_strategy == "Mixed":
                    for i in range(num_tickets):
                        strategy = strategies[i % len(strategies)]
                        ticket = generate_ticket(strategy, main_freq, markov_matrix, data)
                        if exclude_past_winners:
                            while not check_ticket(ticket, full_data).empty:
                                ticket = generate_ticket(strategy, main_freq, markov_matrix, data)
                        main, pb = ticket
                        generated_tickets.append((', '.join(map(str, main)), pb, strategy))
                else: # For single-strategy generation
                    for _ in range(num_tickets):
                        ticket = generate_ticket(gen_strategy, main_freq, markov_matrix, data)
                        if exclude_past_winners:
                            while not check_ticket(ticket, full_data).empty:
                                ticket = generate_ticket(gen_strategy, main_freq, markov_matrix, data)
                        main, pb = ticket
                        generated_tickets.append((', '.join(map(str, main)), pb, gen_strategy))

            df_generated = pd.DataFrame(
                generated_tickets, 
                columns=["Main Numbers", "Powerball", "Strategy"],
                index=np.arange(1, len(generated_tickets) + 1) # <-- FIX: 1-based index
            )
            df_generated.index.name = "Ticket"
            st.session_state.generated_numbers = df_generated

        if 'generated_numbers' in st.session_state:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Your Generated Tickets:")
            st.sidebar.dataframe(st.session_state.generated_numbers)
        
        # --- Frequency Analysis Section ---
        hot_numbers, cold_numbers = get_hot_cold_numbers(main_freq)
        
        st.header("📊 Number Frequency Analysis")
        st.markdown("Explore which numbers have been drawn most and least frequently over the dataset's history.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("🔥 Hot Numbers")
            st.markdown("Most frequently drawn main numbers.")
            for num in hot_numbers:
                st.metric(label=f"Number {num}", value=f"{main_freq[num]} times")

        with col2:
            st.subheader("🧊 Cold Numbers")
            st.markdown("Least frequently drawn main numbers.")
            for num in cold_numbers:
                st.metric(label=f"Number {num}", value=f"{main_freq.get(num, 0)} times")
        
        with col3:
            st.subheader("🔴 Powerball")
            st.markdown("Most frequent Powerball numbers.")
            hot_pb = [num for num, _ in powerball_freq.most_common(5)]
            for pb in hot_pb:
                st.metric(label=f"Powerball {pb}", value=f"{powerball_freq[pb]} times")

        # --- Frequency Charts ---
        freq_df = pd.DataFrame(main_freq.items(), columns=['Number', 'Count']).sort_values('Count', ascending=False)
        
        chart = alt.Chart(freq_df).mark_bar().encode(
            x=alt.X('Number:O', sort=None),
            y=alt.Y('Count:Q'),
            tooltip=['Number', 'Count']
        ).properties(
            title="Frequency of Main Numbers (1-69)"
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

        # --- Statistical Breakdowns ---
        st.header("📈 Statistical Breakdowns")
        st.markdown("Analysis of common patterns in the winning numbers.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Odd vs. Even Numbers")
            odd_even_counts = analyze_odd_even(data)
            odd_even_df = pd.DataFrame({
                'Combination': [f"{i} Odd, {5-i} Even" for i in odd_even_counts.index],
                'Count': odd_even_counts.values
            })
            pie_chart_odd_even = alt.Chart(odd_even_df).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Combination", type="nominal"),
                tooltip=['Combination', 'Count']
            ).properties(title="Odd/Even Combinations")
            st.altair_chart(pie_chart_odd_even, use_container_width=True)

        with col2:
            st.subheader("Low vs. High Numbers")
            low_high_counts = analyze_low_high(data)
            low_high_df = pd.DataFrame({
                'Combination': [f"{i} Low, {5-i} High" for i in low_high_counts.index],
                'Count': low_high_counts.values
            })
            pie_chart_low_high = alt.Chart(low_high_df).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Combination", type="nominal"),
                tooltip=['Combination', 'Count']
            ).properties(title="Low/High Combinations")
            st.altair_chart(pie_chart_low_high, use_container_width=True)

        # --- Randomness Analysis ---
        st.header("🔬 Is it Truly Random?")
        st.markdown(
            "Here, we use a **Chi-Squared Goodness of Fit Test** to see if the observed "
            "number frequencies are consistent with what we'd expect from a perfectly "
            "random lottery. A high p-value suggests the variations are normal; a very "
            "low p-value could suggest a bias."
        )

        # Main Numbers Test
        observed_main_freqs = list(main_freq.values())
        total_main_draws = sum(observed_main_freqs)
        expected_main_freq = total_main_draws / 69
        
        # Create the contingency table
        observed_main = np.array(observed_main_freqs)
        expected_main = np.full_like(observed_main, fill_value=expected_main_freq)
        
        chi2_main, p_main, _, _ = chi2_contingency([observed_main, expected_main])
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Main Numbers (1-69)")
            st.metric("Chi-Squared Statistic", f"{chi2_main:.2f}")
            st.metric("P-value", f"{p_main:.4f}")
            if p_main < 0.05:
                st.warning("The p-value is very low, suggesting the observed frequencies are statistically different from a perfect random distribution.")
            else:
                st.success("The p-value is high, suggesting the frequency variations are consistent with random chance.")

        # Powerball Test
        observed_pb_freqs = list(powerball_freq.values())
        total_pb_draws = sum(observed_pb_freqs)
        expected_pb_freq = total_pb_draws / 26

        observed_pb = np.array(observed_pb_freqs)
        expected_pb = np.full_like(observed_pb, fill_value=expected_pb_freq)

        chi2_pb, p_pb, _, _ = chi2_contingency([observed_pb, expected_pb])
        
        with col2:
            st.subheader("Powerball (1-26)")
            st.metric("Chi-Squared Statistic", f"{chi2_pb:.2f}")
            st.metric("P-value", f"{p_pb:.4f}")
            if p_pb < 0.05:
                st.warning("The p-value is very low, suggesting a potential bias in the Powerball drawings.")
            else:
                st.success("The p-value is high, suggesting the Powerball frequency variations are consistent with random chance.")

        # --- Markov Chain Sandbox ---
        st.header("🎲 Markov Chain Sandbox")
        st.warning(
            "**Disclaimer:** This model is for educational and entertainment purposes only. "
            "Lottery drawings are independent, random events. This model has **no real predictive power** "
            "and treats the numbers as if they were dependent on each other, which they are not."
        )

        st.markdown(
            "The 'Markov' and 'Mixed' options in the sidebar's Number Generation Suite use this model. "
            "Below is the transition matrix showing the probability of the next number's bin, given the current number's bin."
        )
        st.dataframe(markov_matrix.style.format("{:.2%}"))

        # --- Ticket Checker ---
        st.header("🎟️ Check Your Ticket")
        st.markdown("Enter your ticket numbers to see if they've ever won, or upload a file of tickets.")
        
        # --- Draw Date Selector ---
        st.subheader("Select Drawing to Check Against")
        available_dates = full_data.index.strftime('%Y-%m-%d').tolist()
        selected_date_str = st.selectbox(
            "Drawing Date:",
            options=available_dates,
            index=0, # Default to the most recent draw
            help="Defaults to the latest drawing in the dataset."
        )
        selected_date = pd.to_datetime(selected_date_str)
        winning_draw = full_data.loc[selected_date]
        
        st.info(
            f"Checking against the draw on **{selected_date_str}**. "
            f"Winning Numbers: **{', '.join(map(str, winning_draw['Main Numbers']))} | Powerball: {winning_draw['Powerball']}**"
        )

        # --- Single Ticket Checker ---
        st.subheader("Check a Single Ticket")
        ticket_input = st.text_input(
            "Enter your 5 main numbers and 1 Powerball number, separated by commas",
            "1, 2, 3, 4, 5, 6"
        )
        if st.button("Check My Ticket"):
            try:
                numbers = [int(n.strip()) for n in ticket_input.split(',')]
                if len(numbers) == 6:
                    user_ticket = (sorted(numbers[:-1]), numbers[-1])
                    
                    tier, amount = calculate_winnings(user_ticket, winning_draw)
                    
                    if amount > 0:
                        st.balloons()
                        st.success(f"🎉 Congratulations! You won! 🎉")
                        st.metric(label=f"Prize Tier: {tier}", value=f"${amount:,.2f}")
                    else:
                        st.info("Not a winner for this drawing. Better luck next time!")
                else:
                    st.warning("Please enter 6 numbers.")
            except ValueError:
                st.error("Invalid input. Please enter numbers separated by commas.")
                
        # --- File Uploader ---
        st.subheader("Upload and Check Multiple Tickets")
        uploaded_file = st.file_uploader("Upload a .txt or .csv file with your tickets", type=["txt", "csv"])
        
        if uploaded_file is not None:
            tickets_to_check = []
            try:
                # Handle CSV files
                if uploaded_file.name.endswith('.csv'):
                    df_upload = pd.read_csv(uploaded_file)
                    # Check for expected columns
                    if 'Main Numbers' in df_upload.columns and 'Powerball' in df_upload.columns:
                        for _, row in df_upload.iterrows():
                            main_str = str(row['Main Numbers'])
                            main_nums = [int(n.strip()) for n in main_str.split(',')]
                            pb = int(row['Powerball'])
                            if len(main_nums) == 5:
                                tickets_to_check.append((sorted(main_nums), pb))
                    else:
                        st.warning("CSV file must contain 'Main Numbers' and 'Powerball' columns.")

                # Handle TXT files
                elif uploaded_file.name.endswith('.txt'):
                    for line in uploaded_file:
                        line_str = line.decode('utf-8').strip()
                        if not line_str: continue # skip empty lines
                        numbers = [int(n.strip()) for n in line_str.split(',')]
                        if len(numbers) == 6:
                            tickets_to_check.append((sorted(numbers[:-1]), numbers[-1]))
            except Exception as e:
                st.error(f"Error processing file: {e}")

            
            if tickets_to_check:
                all_winnings = []
                with st.spinner(f"Checking {len(tickets_to_check)} tickets against the {selected_date_str} draw..."):
                    for ticket in tickets_to_check:
                        tier, amount = calculate_winnings(ticket, winning_draw)
                        if amount > 0:
                            all_winnings.append({
                                "Your Ticket": f"{', '.join(map(str, ticket[0]))} | PB: {ticket[1]}",
                                "Winning Draw Date": selected_date.date(),
                                "Prize Tier": tier,
                                "Estimated Winnings": amount
                            })
                
                if all_winnings:
                    st.balloons()
                    st.success("🎉 Congratulations! You have winning tickets.")
                    winnings_df = pd.DataFrame(all_winnings)
                    st.dataframe(winnings_df)
                    total_winnings = winnings_df["Estimated Winnings"].sum()
                    st.metric("Total Estimated Winnings", f"${total_winnings:,.2f}")
                else:
                    st.info("No winning tickets found in the uploaded file.")


        st.header("Recent Winning Numbers")
        st.dataframe(data.head())

    else:
        st.error("Could not load or process Powerball data. Please check the data source or try again later.")


if __name__ == "__main__":
    main() 
