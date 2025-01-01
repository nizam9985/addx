import streamlit as st

# Set the title of the app
st.title("Simple Addition App")

# Input fields for numbers
num1 = st.number_input("Enter the first number:", value=0.0, step=1.0, format="%.2f")
num2 = st.number_input("Enter the second number:", value=0.0, step=1.0, format="%.2f")

# Button to calculate the sum
if st.button("Calculate Sum"):
    result = num1 + num2
    st.success(f"The sum of {num1} and {num2} is {result}")