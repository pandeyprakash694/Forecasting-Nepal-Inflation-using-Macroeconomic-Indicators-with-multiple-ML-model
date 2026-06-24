import pandas as pd
import numpy as np
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio


smi = pd.read_csv(os.path.join(os.path.dirname(__file__), '1_SMIs.csv'))
print(smi.head())
