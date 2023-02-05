import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

coins = [
    'ldo',
]


def inflation(df, emmission_schedule):

    if emmission_schedule == '30D':
        d = 12
    elif emmission_schedule == '7D':
        d = 52
    elif emmission_schedule == '1D':
        d = 365

    l = [0]

    total = df.total.to_numpy()
    diff = df.new_supply.to_numpy()

    for i in range(1, len(df)):
        l.append((diff[i] / total[i - 1] * d))

    return np.array(l) * 100


@st.cache(suppress_st_warning=True)
def read_data():

    urls = [
        f'https://raw.githubusercontent.com/kodi1453/jarvis/main/data/{coin}.csv'
        for coin in coins
    ]

    data_dict = {}
    dist_dict = {}
    supply_dict = {}
    totalsupply_dict = {}
    parties_dict = {}

    for u in urls:

        # take file name
        f0 = os.path.basename(u)
        f1 = f0.rsplit('.')[0].upper()

        ## READ DATA
        # general token data
        df_data = pd.read_csv(u, skiprows=[0], nrows=7).set_index('key')
        dict_data = df_data['data'].to_dict()

        # data related to token distribution among different parties
        df_distribution = pd.read_csv(u, skiprows=np.linspace(0, 9, 10))

        # supply for entities based on data from above
        supply = df_distribution.iloc[5:]
        dates = pd.date_range(
            start=dict_data['start_date'],
            periods=len(supply),
            freq=dict_data['emission_schedule'],
        )  # calculate datetime
        supply = supply.set_index(dates).drop(
            columns='entity'
        )  # set index as date
        supply = supply.astype(float)

        # classify entities
        df_distribution = df_distribution.drop(columns='entity')

        team = df_distribution.iloc[1] == 'team_advisors'
        investors = df_distribution.iloc[1] == 'investor'
        public = df_distribution.iloc[1] == 'public'
        foundation = df_distribution.iloc[1] == 'foundation'
        ecosystem = df_distribution.iloc[1] == 'ecosystem'
        validators = df_distribution.iloc[1] == 'validators'

        parties = [team, investors, public, foundation, ecosystem, validators]

        team_df = supply[supply.columns[team.values]]
        investors_df = supply[supply.columns[investors.values]]
        public_df = supply[supply.columns[public.values]]
        foundation_df = supply[supply.columns[foundation.values]]
        ecosystem_df = supply[supply.columns[ecosystem.values]]
        validators_df = supply[supply.columns[validators.values]]

        # total supply
        totalsupply = pd.concat(
            [
                supply,
                team_df.sum(axis=1),
                investors_df.sum(axis=1),
                public_df.sum(axis=1),
                foundation_df.sum(axis=1),
                ecosystem_df.sum(axis=1),
                validators_df.sum(axis=1),
            ],
            axis=1,
        )
        totalsupply = totalsupply.rename(
            columns={
                0: 'team_advisors',
                1: 'investors',
                2: 'public',
                3: 'foundation',
                4: 'ecosystem',
                5: 'validators',
            }
        )

        totalsupply['total'] = supply.sum(axis=1)
        totalsupply['new_supply'] = totalsupply.total.diff()
        totalsupply['annual_inflation'] = inflation(
            totalsupply, dict_data['emission_schedule']
        )

        # create dict entry with dataframe as value and filename as key
        data_dict[f1] = dict_data
        dist_dict[f1] = df_distribution
        supply_dict[f1] = supply
        totalsupply_dict[f1] = totalsupply

    return data_dict, dist_dict, supply_dict, totalsupply_dict


def final_distro_pie(data, labels, token, hoverinfo, textinfo):

    # plotly pie chart of final token distribution
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=data,
            )
        ]
    )
    fig.update_traces(
        hoverinfo=hoverinfo, textinfo=textinfo
    )  # remove to show only % label on chart
    fig.update_layout(title_text=f'{token} Allocation', title_x=0.45)

    return fig


def supply_distribution_area(data, token):

    # stacked area chart of supply distribution
    fig = px.area(data)
    fig.update_layout(
        title_text=f'{token} Supply Distribution',
        title_x=0.45,
        xaxis_title='Date',
        yaxis_title='Token Supply',
        legend_title='Holders',
        plot_bgcolor='rgba(0, 0, 0, 0)',
    )

    return fig


def evolution_supply_dist(data, token):

    # Evolution of supply distribution %
    fig = px.area(
        data,
        title=f'{token} Supply %',
        groupnorm='fraction',
    )
    fig.update_layout(
        yaxis=dict(showgrid=True, gridcolor='rgba(166, 166, 166, 0.35)'),
        yaxis_tickformat=',.0%',
        xaxis=dict(showgrid=False, gridcolor='rgba(166, 166, 166, 0.35)'),
        xaxis_title='Date',
        yaxis_title='Token Supply',
        legend_title='Holders',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        autosize=False,
        width=1000,
        height=550,
        title_x=0.4,
    )

    return fig


def main():

    (
        data_dict,
        dist_dict,
        supply_dict,
        totalsupply_dict,
        parties_dict,
    ) = read_data()

    st.title('Lido Supply Distribution')

    token = st.sidebar.selectbox(
        'What token do you want to know more about?',
        [coin.upper() for coin in coins],
    )

    all_initial_allo = (dist_dict[token].iloc[3]).astype(float)

    # plotly pie chart of final token distribution
    fig = final_distro_pie(
        data=all_initial_allo.values,
        labels=all_initial_allo.index,
        token=token,
        hoverinfo='label+percent',
        textinfo='label+percent',
    )
    st.plotly_chart(fig, use_container_width=True)

    # stacked area chart of supply distribution
    fig = supply_distribution_area(
        data=totalsupply_dict[token].drop(
            columns=[
                'team_advisors',
                'investors',
                'public',
                'foundation',
                'ecosystem',
                'validators',
                'annual_inflation',
                'total',
                'new_supply',
            ]
        ),
        token=token,
    )
    st.plotly_chart(fig, use_container_width=True)

    # fractional evolution of supply distribution
    fig = evolution_supply_dist(
        data=totalsupply_dict[token].drop(
            columns=[
                'team_advisors',
                'investors',
                'public',
                'foundation',
                'ecosystem',
                'validators',
                'annual_inflation',
                'total',
                'new_supply',
            ]
        ),
        token=token,
    )
    st.plotly_chart(fig, use_container_width=True)


    # inflation charts
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=totalsupply_dict[token].index,
            y=totalsupply_dict[token]['new_supply'],
            name=f'{token} New Monthly Supply',
        )
    )
    fig.add_trace(
        go.Scatter(
            x=totalsupply_dict[token].index,
            y=totalsupply_dict[token]['annual_inflation'],
            name=f'{token} Annual Inflation',
            yaxis='y2',
        )
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(
        yaxis=dict(
            title=f'{token} New Supply [%]',
            titlefont=dict(color='#1f77b4'),
            tickfont=dict(color='#1f77b4'),
            autotypenumbers='convert types',
            showgrid=True,
            gridcolor='rgba(166, 166, 166, 0.35)',
        ),
        yaxis2=dict(
            title=f'{token} Annual Inflation [%]',
            titlefont=dict(color='#d62728'),
            tickfont=dict(color='#d62728'),
            anchor='x',
            overlaying='y',
            side='right',
        ),
        showlegend=False,
        plot_bgcolor='white',
        autosize=False,
        width=800,
        height=550,
        title_x=0.5,
    )

    st.plotly_chart(fig, use_container_width=True)

    return


if __name__ == '__main__':
    st.set_page_config(page_title='Supply Distribution Tracker')
    main()