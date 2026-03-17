import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


def score_distribution(df):

    fig = px.histogram(
        df,
        x="math_score",
        nbins=30,
        title="Math Score Distribution",
        color_discrete_sequence=["#69AEF8"]
    )

    return fig


def gender_performance(df):

    fig = px.box(
        df,
        x="gender",
        y="math_score",
        color="gender",
        title="Gender vs Math Score"
    )

    return fig


def lunch_vs_score(df):

    fig = px.box(
        df,
        x="lunch",
        y="math_score",
        color="lunch",
        title="Lunch Type vs Score"
    )

    return fig


def parental_education(df):

    fig = px.bar(
        df,
        x="parental_level_of_education",
        y="math_score",
        color="parental_level_of_education",
        title="Parental Education vs Math Score"
    )

    return fig


def correlation_heatmap(df):

    plt.figure(figsize=(8,6))

    sns.heatmap(
        df[
            ["math_score","reading_score","writing_score"]
        ].corr(),
        annot=True,
        cmap="cividis"
    )

    plt.title("Correlation Heatmap")

    return plt