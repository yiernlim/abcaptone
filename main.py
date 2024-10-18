import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
import json
import lolviz
import re
import matplotlib.pyplot as plt
import random
import hmac
from utility import check_password

# Import the key CrewAI classes
from crewai import Agent, Task, Crew
from crewai_tools import WebsiteSearchTool

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# load environment variables from .env file
#load_dotenv('.env')

# Retrieve the OpenAI API key from the environment variables
#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai.api_key)

# Define CPF Interest Rates
CPF_INTEREST_OA = 0.025  # Ordinary Account Interest Rate (2.5%)
CPF_INTEREST_SA = 0.04   # Special Account Interest Rate (4%)
CPF_INTEREST_MA = 0.04   # Medisave Account Interest Rate (4%)

# Create the website search tool (searches within CPF's domain)
tool_websearch = WebsiteSearchTool("https://cpf.gov.sg/")

# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()

# Sanitize user input to prevent prompt injection attacks
def sanitize_input(user_input):
    # Limit input length (max 500 characters for example)
    user_input = user_input[:500]

    # Remove dangerous symbols and characters
    user_input = re.sub(r"[{};$]", "", user_input)

    # Validate using a regex to allow only safe alphanumeric characters and common punctuation
    sanitized_input = re.sub(r"[^\w\s\.\,\?\!]", "", user_input)
    
    return sanitized_input.strip()

# This a new helper
# Note that this function directly take in "messages" as the parameter.
def get_completion_by_messages(messages, model="gpt-4o-mini", temperature=0, top_p=1.0, max_tokens=1024, n=1):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1
    )
    return response.choices[0].message.content

# create agents for policy explainer
def create_agents(user_query):

    user_query = sanitize_input(user_query)
    
    # Agent 1: Content Planner (Plans the simplified content based on user query)
    agent_planner = Agent(
        role="Content Planner",
        goal=f"Plan content to simplify the CPF policy based on '{user_query}'",
        backstory=f"You're tasked with creating a plan to simplify '{user_query}'. Your goal is to structure the response in a way that's easy for users to understand, and to guide other agents (such as researchers) in gathering information.",
        allow_delegation=True,
        verbose=True,
    )

    # Agent 2: Research Analyst (Researches CPF policies)
    agent_researcher = Agent(
        role="Research Analyst",
        goal=f"Conduct in-depth research on '{user_query}' and provide necessary information to the Content Planner",
        backstory=f"You're responsible for conducting research on CPF policies based on '{user_query}'. Use the web search tool to gather information from reliable sources such as the CPF official website. Avoid using LaTeX symbols or complex mathematical notations. Keep your research findings in plain text and easy-to-understand format.",
        allow_delegation=False,
        verbose=True,
    )

    # Agent 3: Content Writer (Simplifies the CPF policy explanation)
    agent_writer = Agent(
        role="Content Writer",
        goal=f"Write a simplified explanation of '{user_query}' for the user",
        backstory=f"Based on the research and the plan from the Content Planner, your goal is to create a user-friendly, easy-to-understand explanation of '{user_query}'. Avoid using LaTeX symbols or complex mathematical notations, and ensure the explanation is delivered in plain text.",
        allow_delegation=False,
        verbose=True,
    )

    return agent_planner, agent_researcher, agent_writer

# create tasks for policy explainer
def create_tasks(agent_planner, agent_researcher, agent_writer, user_query):
    task_plan = Task(
        description=f"""\
        1. Plan the structure and outline of the simplified explanation.
        2. Identify key elements of the CPF policy based on the '{user_query}'.""",
        expected_output="A content plan and outline for explaining the CPF policy.",
        agent=agent_planner,
    )

    task_research = Task(
        description=f"""\
        1. Conduct research on the '{user_query}' queried by the user.
        2. Gather information from CPF's official website and other trusted sources.
        3. Ensure the research is explained clearly, without using LaTeX symbols or complex mathematical notations.""",
        expected_output="A detailed research report containing the key insights on the CPF policy, written in plain text.",
        agent=agent_researcher,
        tools=[tool_websearch],
    )

    task_write = Task(
        description="""\
        1. Write a simplified, easy-to-understand explanation based on the research.
        2. Ensure the explanation is clear, concise, and includes practical examples.
        3. Avoid using LaTeX symbols or complex mathematical notations in the final explanation.""",
        expected_output="A simplified CPF policy explanation ready for user consumption, in plain text without complex formatting.",
        agent=agent_writer,
    )

    return task_plan, task_research, task_write

# Create agents for retirement planning
def create_retirement_planner_agents(user_query):
    user_query = sanitize_input(user_query)
    
    agent_planner = Agent(
        role="Retirement Planner",
        goal=f"Plan content to help users understand how to save for retirement based on '{user_query}'",
        backstory=f"You're responsible for planning the retirement strategy based on '{user_query}'. Ensure the explanation is simple, free from any mathematical symbols, formulas, or LaTeX, and easy to understand.",
        allow_delegation=True,
        verbose=True,
    )

    agent_researcher = Agent(
        role="Retirement Research Analyst",
        goal=f"Conduct in-depth research on '{user_query}' and provide necessary information to the Retirement Planner",
        backstory=f"You're responsible for researching CPF retirement planning and best practices based on '{user_query}'. Use the web search tool to gather information from reliable sources such as the CPF official website. Ensure the research findings are written in plain text, avoiding LaTeX symbols, formulas, or complex mathematical notations.",
        allow_delegation=False,
        verbose=True,
    )

    agent_writer = Agent(
        role="Financial Advisor",
        goal=f"Write financial advice on how to improve retirement savings and extend savings based on '{user_query}'. The advice should be easy to understand, free from LaTeX symbols or formulas, and focused on practical advice written in plain language.",
        backstory=f"Based on the research and plan, create an easy-to-understand explanation of '{user_query}'. Avoid using LaTeX symbols, complex mathematical notations, or formulas. Focus on providing practical advice using plain language and basic arithmetic.",
        allow_delegation=False,
        verbose=True,
    )

    return agent_planner, agent_researcher, agent_writer

# create tasks for retirement planning
def create_retirement_planner_tasks(agent_planner, agent_researcher, agent_writer, user_query):
    task_plan = Task(
        description=f"Plan the structure and steps for retirement planning based on '{user_query}'. Ensure the plan is easy to follow, written in plain text, and avoids all mathematical symbols or LaTeX.",
        expected_output="A detailed plan for retirement without complex symbols or LaTeX.",
        agent=agent_planner,
    )

    task_research = Task(
        description=f"Research CPF savings growth, tax benefits, and strategies to maximize savings based on '{user_query}'. Ensure the research report is written in plain text and avoids LaTeX symbols, formulas, or complex mathematical notations.",
        expected_output="A detailed research report on CPF retirement planning, written in simple language, avoiding any mathematical symbols or formulas.",
        agent=agent_researcher,
    )

    task_write = Task(
        description=f"Write financial advice to help users extend their CPF savings for retirement. The advice should be easy to understand, free from LaTeX symbols or formulas, and focused on practical advice written in plain language.",
        expected_output="A detailed financial advice report written in plain text and easily understandable by users, avoiding any LaTeX symbols or complex mathematical notations.",
        agent=agent_writer,
    )

    return task_plan, task_research, task_write


# Function for CPF Policy Explainer page
def cpf_policy_explainer():
    st.header("CPF Policy Explainer")
    
    # Create a form with a text input to allow user to ask questions about CPF policies
    with st.form(key="cpf_form"):
        user_query = st.text_area("Enter your CPF Policy related question.", placeholder="E.g., How does Medisave work?")
        submit_btn = st.form_submit_button("Submit Query")
    
    if submit_btn and user_query:
        user_query = sanitize_input(user_query)
        try:
            with st.spinner('Generating your explanation...'):
                # Create agents and tasks dynamically based on user query
                agent_planner, agent_researcher, agent_writer = create_agents(user_query)
                task_plan, task_research, task_write = create_tasks(agent_planner, agent_researcher, agent_writer, user_query)

                # Create the crew (team of agents) responsible for the task
                crew = Crew(
                    agents=[agent_planner, agent_researcher, agent_writer],
                    tasks=[task_plan, task_research, task_write],
                    verbose=True,
                )
                
                # Execute the crew's tasks based on the user query
                result = crew.kickoff(inputs={"topic": user_query})
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return
        
        # Display the entire result to understand its structure
        # st.subheader("Raw Output from Crew")
        # st.write(result)  # Show the raw result structure for inspection

        # Access the tasks_output from the result (assuming result.tasks_output is a list)
        tasks_output = result.tasks_output  # Directly access tasks_output

        # Initialize variable to store the final answer (from the Content Writer)
        final_answer = None
        
        # Loop through the tasks_output to find the result from the 'Content Writer' agent
        for task_output in tasks_output:
            if task_output.agent == 'Content Writer':  # Check if the agent is the Content Writer
                final_answer = task_output.raw  # Extract the 'raw' output from the Content Writer's task
                break  # Once found, we can exit the loop

        # Display the final answer if we found it
        if final_answer:
            st.subheader("Simplified Explanation")
            st.markdown(final_answer)  # Display the final answer using markdown
        else:
            st.write("No final explanation could be generated for the query.")


# Function for retirement planner with prompt chaining and prompt engineering
def retirement_planner():
    st.subheader("CPF Retirement Planning Simulator")
    st.write("This feature will allow users to simulate their retirement planning.")
    
    # Step 1: User Inputs
    with st.form(key="retirement_form"):
        current_age = st.number_input("Enter your current age", min_value=18, max_value=100, value=30)
        retirement_age = st.number_input("Enter your desired retirement age", max_value=70, value=65)
        current_oa = st.number_input("Current CPF Ordinary Account (OA) savings", value=50000)
        current_sa = st.number_input("Current CPF Special Account (SA) savings", value=20000)
        current_ma = st.number_input("Current CPF Medisave Account (MA) savings", value=10000)
        monthly_oa_contribution = st.number_input("Monthly CPF OA contribution", value=1000)
        monthly_sa_contribution = st.number_input("Monthly CPF SA contribution", value=500)
        monthly_ma_contribution = st.number_input("Monthly CPF MA contribution", value=300)
        desired_monthly_payout = st.number_input("Desired monthly payout after retirement", value=2000)
        user_query = st.text_area("Any specific retirement goals or questions?", placeholder="E.g., How is CPF savings related to retirement?")
        submit_btn = st.form_submit_button("Submit")

    if submit_btn and user_query:
        user_query = sanitize_input(user_query)

        # Step 2: Agent Creation and Task Execution (Retirement Planning Use Case)
        with st.spinner("Planning your retirement..."):
            try:
                agent_planner, agent_researcher, agent_writer = create_retirement_planner_agents(user_query)
                task_plan, task_research, task_write = create_retirement_planner_tasks(agent_planner, agent_researcher, agent_writer, user_query)

                crew = Crew(
                    agents=[agent_planner, agent_researcher, agent_writer],
                    tasks=[task_plan, task_research, task_write],
                    verbose=True,
                )
                
                # Simulate the result after agents' tasks are executed
                result = crew.kickoff(inputs={"topic": user_query})
                
                # Get explanation from Writer Agent
                tasks_output = result.tasks_output
                final_answer = None
                
                for task_output in tasks_output:
                    if task_output.agent == 'Financial Advisor':
                        final_answer = task_output.raw
                        break
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                return

        # Step 3: Visualizing Retirement Growth and Payouts
        years_to_retirement = retirement_age - current_age
        oa_savings, sa_savings, ma_savings = current_oa, current_sa, current_ma
        oa_growth, sa_growth, ma_growth = [], [], []

        for year in range(years_to_retirement):
            oa_savings += monthly_oa_contribution * 12
            sa_savings += monthly_sa_contribution * 12
            ma_savings += monthly_ma_contribution * 12

            oa_savings *= (1 + CPF_INTEREST_OA)
            sa_savings *= (1 + CPF_INTEREST_SA)
            ma_savings *= (1 + CPF_INTEREST_MA)

            oa_growth.append(oa_savings)
            sa_growth.append(sa_savings)
            ma_growth.append(ma_savings)

        total_savings = oa_savings + sa_savings + ma_savings
        st.subheader(f"Total CPF Savings at Retirement: ${total_savings:,.2f}")

        months_of_payout = total_savings // desired_monthly_payout
        years_of_payout = months_of_payout // 12
        remaining_months = months_of_payout % 12

        st.subheader(f"Your CPF savings will last for {years_of_payout} years and {remaining_months} months with a monthly payout of ${desired_monthly_payout:,.2f}.")

        # **Visualization of Growth Over Time**
        st.subheader("CPF Account Growth Over Time")
        
        # Visualization Section
        fig, ax = plt.subplots(figsize=(5, 3))
        years = list(range(current_age, retirement_age))  # X-axis representing years

        ax.plot(years, oa_growth, label='Ordinary Account (OA)', color='blue')
        ax.plot(years, sa_growth, label='Special Account (SA)', color='green')
        ax.plot(years, ma_growth, label='Medisave Account (MA)', color='red')

        ax.set_xlabel('Age')
        ax.set_ylabel('Savings ($)')
        ax.set_title('CPF Account Growth Until Retirement')
        ax.legend()

        st.pyplot(fig)  # Display the plot in Streamlit

        # Step 4: Explanation via LLM
        explainer_prompt = f"""
        I have ${total_savings:,.2f} saved for retirement at age {retirement_age}.
        I want to withdraw ${desired_monthly_payout:,.2f} per month after retirement.
        Please provide advice on how long my savings will last and suggestions for extending my savings. Note that Ordinary Account Interest Rate is 2.5%, Special Account Interest Rate and Medisave Account Interest Rate are 4%.
        Provide multiple scenarios where the user adjusts their retirement plans (e.g., different monthly payouts, later retirement age, or additional savings).
        Avoid using LaTeX or complex mathematical notations. Provide your answer in plain text, using simple terms for calculations.
        """

        messages = [
            {"role": "system", "content": "You are a financial advisor specializing in retirement planning."},
            {"role": "user", "content": explainer_prompt}
        ]

         # Get LLM response using helper function
        explanation = get_completion_by_messages(messages)

        # Display Financial Advice first
        st.subheader("Financial Advice")
        st.write(explanation)

        # Display Answer to User's Query (Final Answer from Writer Agent)
        if final_answer:
            st.subheader("Answer to your Query")
            st.markdown(final_answer)
        else:
            st.write("No final explanation could be generated for the query.")

def about_us():
    st.write("""
    # About Us

    Welcome to **CPF Policy Explainer & Retirement Planning Simulator**. This web application has been developed as part of a capstone assignment to help Singaporean citizens seamlessly interact with publicly available information related to CPF (Central Provident Fund) policies and retirement planning. It consolidates reliable data from official sources, simplifies complex information, and provides personalised simulations and guidance.

    ## Project Scope
    The objective of this project is to create a user-friendly web-based application that enables Singaporeans to easily access and understand CPF-related policies, while also providing interactive inputs to simulate retirement scenarios. By aggregating data from multiple trustworthy sources, the application allows users to explore how their CPF savings, retirement age, and other financial decisions impact their retirement planning.

    ## Objectives
    - **Increase Understanding**: Helps to simplify and explain CPF policies based on user query.
    - **Empower Decision Making**: Provide personalised retirement planning to help users make informed financial decisions.
    - **Interactive Learning**: Allow users to explore different retirement scenarios with real data and forecasts.


    ## Data Sources
    This application utilizes data from the following trusted and official sources:
    - **Central Provident Fund Board (CPF)**: Official Website: policies, contribution rates, and CPF interest rates.
    - **Singapore Government Websites**: Data on retirement age, retirement-related policies, and government-backed retirement schemes.
    - **Statistical Data**: Latest data on CPF interest rates, contribution limits, and relevant financial figures.

    ## Features
    ### 1. CPF Policy Explainer
    The CPF Policy Explainer provides an easy-to-understand breakdown of complex CPF policies. It allows users to ask questions and receive simplified explanations. This feature includes:
    
    - **Interactive Queries**: Users can input their queries related to CPF policies, such as Medisave, housing, or retirement, and receive tailored, easy-to-understand answers.
    - **Comprehensive Research**: Behind the scenes, the system gathers relevant information from official CPF sources to ensure accurate, reliable explanations.
    - **Plain Text Explanations**: Complex terms, mathematical notations, and jargon are avoided to ensure that the policy details are presented in a clear and simple manner.
    
    ### 2. Retirement Planning Simulator
    The Retirement Planning Simulator allows users to simulate various retirement scenarios based on their CPF savings, expected retirement age, and financial goals. Users can see how their savings grow over time and estimate how long their savings will last after retirement based on different monthly payouts. The simulator offers:

    - **Customizable Input Fields**: Users can input their current age, desired retirement age, CPF savings (OA, SA, MA), and monthly contributions to simulate their retirement scenario.
    - **Real-Time Visualizations**: Graphical representations of CPF savings growth until retirement.
    - **Personalized Projections**: Tailored financial advice and scenarios, helping users explore how adjustments to their savings or retirement age can extend their savings.
   

    ## Possible Future Developments
    In future iterations of this application, the following improvements can be made:
    - **Comparison Tools**: Offer tools that allow users to compare their CPF savings and retirement plans with the national average or peers in their demographic group, helping users gauge where they stand.
    - **Investment and Savings Projections**: Include tools that allow users to factor in additional investments outside of CPF, such as stock portfolios, savings accounts, or property investments, to get a more holistic view of their financial future.

    """)


def methodology():
    st.write("""
    # Methodology
    This application integrates multiple components to provide an interactive experience for both CPF policy explanations and retirement planning simulations. Below is a comprehensive breakdown of how data flows and the technical implementation details.
    
    ## Overview of Components
    The web application consists of two core features:
    1. **CPF Policy Explainer**: Provides simplified, user-friendly explanations of CPF policies based on user queries.
    2. **Retirement Planning Simulator**: Simulates personalised retirement scenarios using user-provided CPF data.
             
    The web application is built with Python and Streamlit as the frontend framework, and various libraries and APIs are utilised for back-end data processing, sanitization, and visualisation.
             
    ### 1. Data Flow and Implementation details of CPF Policy Explainer 
    - **User Input**: The user submits a query related to CPF policies (e.g how does Medisave work? I am XX age this year.).
    - **Sanitization**: Before processing the input, it goes through a sanitization/handling process using the `sanitize_input()` function implemented. This prevents malicious inputs by removing unwanted symbols and limiting the input length.
    - **Agent-Based System**: 
        - The application uses three agents, each with distinct roles: a Content Planner, a Research Analyst, and a Content Writer. These agents work collaboratively to deliver a simplified explanation.
        - **Agent Workflow**:
            1. **Content Planner**: Structures the query and creates an outline for the explanation.
            2. **Research Analyst**: Gathers detailed information from reliable sources, primarily from the official CPF website using the `WebsiteSearchTool()`.
            3. **Content Writer**: Uses the gathered information to create a user-friendly and simplified explanation.
    - **Task Execution**: Each agent has specific tasks that are created using the `Task` class. The tasks are then managed by the `Crew` object, which orchestrates the execution of tasks to produce a final, coherent explanation.
    - **Output**: The final output from the Content Writer is displayed in plain text which is easily understandable by the user.
             
     ### 2. Data Flow and Implementation details of Retirement Planning Simulator
    - **User Input**: Users provide information such as their current age, CPF savings, desired retirement age, current Ordinary Account (OA), Special Account (SA), and Medisave Account (MA) savings, monthly OA, SA and MA account contribution, desired monthly payout after retirement and their query related to retirement.
    - **Sanitization**: Before processing the input, it goes through a sanitization/handling process using the `sanitize_input()` function implemented. This prevents malicious inputs by removing unwanted symbols and limiting the input length.
    - **Simulation Process**: 
        - The application calculates the growth of the user’s CPF savings based on interest rates for the Ordinary Account (OA), Special Account (SA), and Medisave Account (MA), which are predefined in the system (e.g CPF interest for OA = 2.5%).
        - The calculation process involves compounding the monthly contributions and interest over the years until their specified retirement age.
    - **Agent-Based Planning**: Similar to the Policy Explainer, the simulator also uses an agent-based approach:
        1. **Retirement Planner**: Plans the strategy for saving and retirement based on user inputs.
        2. **Research Analyst**: Gathers information on CPF retirement planning and ensures the savings projections align with CPF regulations and guidelines.
        3. **Financial Advisor**: Provides easy-to-understand financial advice, recommending ways to extend savings and adjust retirement plans if necessary.
    - **Visualization**: The growth of CPF savings is visualised using matplotlib, providing the user with a clear understanding of their financial future.
    - **LLM:** The system formulates a prompt (query) to ask the LLM for financial advice. The system also generates personalised financial advice using an LLM (OpenAI’s GPT model) to explain complex retirement scenarios in plain text. 
    - **2 Final Outputs**: The LLM-generated explanation is displayed as financial advice on the retirement planning page, providing tailored suggestions based on the user's inputs. This helps users understand how long their savings will last and what adjustments they can make to extend their financial security.
        A detailed financial advice to answer user's query related to retirement will also be displayed.
    """)
    # Displaying the flowchart image for CPF Policy Explainer
    st.subheader("3. Flowchart of CPF Policy Explainer")
    policy_explainer_flowchart = os.path.join(os.getcwd(), 'diagrams', 'policy_explainer_flowchart.png')
    if os.path.exists(policy_explainer_flowchart):
        st.image(policy_explainer_flowchart, caption="CPF Policy Explainer Flowchart", use_column_width=True)
    else:
        st.error("Flowchart for CPF Policy Explainer not found.")
    
    # Displaying the flowchart image for Retirement Planning Simulator
    st.subheader("4. Flowchart of Retirement Planning Simulator")

    # Define the path to the flowchart image
    retirement_planning_flowchart = os.path.join(os.getcwd(), 'diagrams', 'retirement_planning_flowchart.png')

    # Check if the image exists and display it
    if os.path.exists(retirement_planning_flowchart):
        st.image(retirement_planning_flowchart, caption="Retirement Planning Simulator Flowchart", use_column_width=True)
    else:
        st.error("Flowchart for Retirement Planning Simulator not found.")

# Set the page configuration
st.set_page_config(
    page_title="My Capstone Assignment", 
    page_icon="🌟",  
)
 
# Title of the application
st.title("CPF Policies Explainer & Retirement Planning Simulator")


# Collapsible disclaimer using st.expander
with st.expander("Disclaimer"):
    st.write("""
    IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.
             Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.
             Always consult with qualified professionals for accurate and personalized advice.
    """)

# Initialize session state if not already initialized
if 'page' not in st.session_state:
    st.session_state.page = "About Us"  # Default page

# Sidebar for navigation with buttons
st.sidebar.title("Navigation")

if st.sidebar.button("About Us"):
    st.session_state.page = "About Us"
if st.sidebar.button("Methodology"):
    st.session_state.page = "Methodology"
if st.sidebar.button("CPF Policy Explainer"):
    st.session_state.page = "CPF Policy Explainer"
if st.sidebar.button("Retirement Planning Simulator"):
    st.session_state.page = "Retirement Planning Simulator"


# Load appropriate page based on user selection (stored in session state)
if st.session_state.page == "About Us":
    about_us()

elif st.session_state.page == "Methodology":
    methodology()

elif st.session_state.page == "CPF Policy Explainer":
    cpf_policy_explainer()

elif st.session_state.page == "Retirement Planning Simulator":
    retirement_planner()