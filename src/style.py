def custom_css():
    return """
    <style>
        /* Import EB Garamond font */
        @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;700&display=swap');

        /* Apply font to all headings */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'EB Garamond', serif !important;
        }

        /* Set background gradient */
        body {
            background: linear-gradient(to bottom right, #ffe0cf, #ffbff6);
        }
        .stApp {
            background: linear-gradient(to bottom right, #ffe0cf, #ffbff6);
        }

        /* Change button styling */
        div.stButton > button {
            background-color: #a48be0 !important; /* Default button color (soft purple) */
            color: white !important; /* Ensures white text */
            font-weight: bold;
            border-radius: 8px;
            border: none;
            transition: background-color 0.3s ease-in-out, color 0.3s ease-in-out;
        }

        /* Button hover effect */
        div.stButton > button:hover {
            background-color: #7a52c7 !important; /* Darker purple when hovered */
            border: 2px solid #5a33a3 !important;
            color: white !important; /* Keep text white */
        }

        /* Button focus (when clicked) */
        div.stButton > button:focus {
            background-color: #7a52c7 !important; /* Keeps button purple on focus */
            border: 2px solid #5a33a3 !important;
            color: white !important;
            outline: none !important;
        }

        /* Button active (when pressed) */
        div.stButton > button:active {
            background-color: #5a33a3 !important; /* Even darker purple when clicked */
            border: 2px solid #3e227a !important;
            color: white !important;
        }


        /* When selected (focused), the border turns purple */
        textarea:focus, input:focus, 
        div[data-baseweb="input"]:focus-within, 
        div[data-baseweb="textarea"]:focus-within {
            border-color: #7a52c7 !important;
            box-shadow: 0px 0px 8px #7a52c7 !important;
            outline: none !important;
        }
    </style>
    """