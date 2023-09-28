# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 22:47:34 2023

@author: mayan_izchdl9
"""

import streamlit as st
from datetime import datetime

def main():
    st.title("Date and Time App")

    # Get current date and time
    current_datetime = datetime.now()
    
    # Format the date and time
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # Display the result
    st.write(f"Today's Date and Time: {formatted_datetime}")

if __name__ == "__main__":
    main()
