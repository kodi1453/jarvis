import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import requests
from etherscan import Etherscan   # wrapper for etherscan api
from web3 import Web3
from typing import Optional
import datetime


class SimpleEtherscanApi:
    """
    Class that leverages Etherscan API.

    Intended as a simplified way of getting transaction and token transfer data.
    It can be scaled in the future.

    Etherscan API Documenation: https://docs.etherscan.io/
    """

    def __init__(
        self,
        etherscan_key: str,
        url: str = 'https://api.etherscan.io/api',
        startblock: int = 0,
        endblock: int = 99999999,
    ) -> None:
        """
        Args:
            etherscan_key (str): your Etherscan API key.
            url (_type_, optional): API endpoint. Defaults to 'https://api.etherscan.io/api'.
            startblock (int, optional): Defaults to 0.
            endblock (int, optional): Defaults to 99999999.
        """
        self.etherscan_key = etherscan_key
        self.etherscan = Etherscan(self.etherscan_key)
        self.url = url
        self.startblock = startblock
        self.endblock = endblock

    def normal_transactions_by_address(
        self, address: str, sort: str = 'asc'
    ) -> pd.DataFrame:
        """
        Return pandas Dataframe of normal transactions for a given address.

        Args:
            address (str): ethereum address to query.
            sort (str, optional): order in which to sort data. Defaults to 'asc'.

        Returns:
            pd.DataFrame: normal transactions
        """
        # API parameters
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': self.startblock,
            'endblock': self.endblock,
            'sort': sort,
            'apikey': self.etherscan_key,
        }

        # Parse the response
        data = json.loads(requests.get(self.url, params=params).text)

        # return only valid transactions
        return pd.DataFrame(
            [tx for tx in data['result'] if tx['isError'] == '0']
        )

    def internal_transactions_by_address(
        self, address: str, sort: str = 'asc'
    ) -> pd.DataFrame:
        """
        Return pandas Dataframe of internal transactions for a given address.

        Args:
            address (str): ethereum address to query.
            sort (str, optional): order in which to sort data. Defaults to 'asc'.

        Returns:
            pd.DataFrame: internal transactions
        """
        # API parameters
        params = {
            'module': 'account',
            'action': 'txlistinternal',
            'address': address,
            'startblock': self.startblock,
            'endblock': self.endblock,
            'sort': sort,
            'apikey': self.etherscan_key,
        }

        # Parse the response
        data = json.loads(requests.get(self.url, params=params).text)

        # return only valid transactions
        return pd.DataFrame(
            [tx for tx in data['result'] if tx['isError'] == '0']
        )

    def erc20_transfers_by_address(
        self,
        sort: str = 'asc',
        address: Optional[str] = None,
        contractaddress: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return pandas Dataframe of ERC20 token transfers for a given address
        and/or contract address.

        Args:
            sort (str, optional): order in which to sort data. Defaults to 'asc'.
            address (Optional[str], optional): Ethereum address to query. Defaults to None.
            contractaddress (Optional[str], optional): contract address to query. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            pd.DataFrame: _description_
        """

        if address and not contractaddress:
            params = {
                'module': 'account',
                'action': 'tokentx',
                'address': address,
                'startblock': self.startblock,
                'endblock': self.endblock,
                'sort': sort,
                'apikey': self.etherscan_key,
            }
        elif contractaddress and not address:
            params = {
                'module': 'account',
                'action': 'tokentx',
                'contractaddress': contractaddress,
                'startblock': self.startblock,
                'endblock': self.endblock,
                'sort': sort,
                'apikey': self.etherscan_key,
            }
        elif address and contractaddress:
            params = {
                'module': 'account',
                'action': 'tokentx',
                'contractaddress': contractaddress,
                'address': address,
                'startblock': self.startblock,
                'endblock': self.endblock,
                'sort': sort,
                'apikey': self.etherscan_key,
            }
        else:
            raise ValueError(
                'You need to specify either an address, a contractaddress, or both.'
            )

        return pd.DataFrame(
            json.loads(requests.get(self.url, params=params).text)['result']
        )


class VestingContract:
    """
    Class for a vesting smart contract that is instantiated using the
    Etherscan and web3 packages.
    """

    def __init__(
        self,
        address: str,
        etherscan_key: str,
        web3_url: str,
        abi: Optional[str] = None,
    ) -> None:
        self.address = address
        self.etherscan_key = etherscan_key
        self.web3_url = web3_url
        # initiate etherscan client
        self.etherscan = Etherscan(self.etherscan_key)
        # initiate web3 client
        self.w3 = Web3(Web3.HTTPProvider(web3_url))
        # get contract abi using etherscan api
        if abi:
            self.abi = abi
        else:
            self.abi = self.etherscan.get_contract_abi(self.address)
        # get checksum address
        self.cheksum = self.w3.toChecksumAddress(self.address)
        # instantiate contract using web3 package
        self.instance = self.w3.eth.contract(
            address=self.cheksum, abi=self.abi
        )
        self.simple_etherscan = SimpleEtherscanApi(self.etherscan_key)

    def list_all_functions(self):
        """
        Returns a list of all functions available for that contract.
        """
        return self.instance.all_functions()

    def list_all_events(self):
        """
        Returns a list of all events possible for that contract.
        """
        return [e for e in self.instance.events]

    def create_event_instance(self, event_name: str):
        """
        Instantiate function event.

        Args:
            event_name (str): name of function event
        """
        return getattr(self.instance.events, event_name)

    def get_all_events(self, event_name: str):
        """
        Returns a list of all past events for a type of event of that contract.
        """
        event_instance = getattr(self.instance.events, event_name)
        return event_instance.createFilter(fromBlock='0x0').get_all_entries()

    def all_events_pretty(self, event_name: str):
        """
        Returns a polished list of all past events for a type of event
        of that contract.
        """
        event_instance = getattr(self.instance.events, event_name)
        entries = event_instance.createFilter(
            fromBlock='0x0'
        ).get_all_entries()
        return entries

    def get_normal_transactions(self):
        """
        Get all normal transactions for contract address.

        Returns:
            pd.Dataframe: normal transactions for contract address
        """
        return self.simple_etherscan.normal_transactions_by_address(
            self.cheksum
        )

    def get_internal_transactions(self):
        """
        Get all internal transactions for contract address.

        Returns:
            pd.Dataframe: internal transactions for contract address
        """
        return self.simple_etherscan.internal_transactions_by_address(
            self.cheksum
        )

    def get_erc20_transfers(
        self,
        sort: str = 'asc',
        address: Optional[str] = None,
        contractaddress: Optional[str] = None,
    ):
        """
        Get all ERC20 transfers for contract address.

        Returns:
            pd.Dataframe: ERC20 transfers for contract address
        """
        return self.simple_etherscan.erc20_transfers_by_address(
            sort=sort, address=address, contractaddress=contractaddress
        )

    def function_by_signature(self, signature):
        return self.instance.get_function_by_signature(signature)

    def function_by_name(self, name):
        return self.instance.get_function_by_name(name)


def inflation(df):

    d = 365
    l = [0]

    total = df.total.to_numpy()
    diff = df.new_supply.to_numpy()

    for i in range(1, len(df)):
        l.append((diff[i] / total[i - 1] * d))

    return np.array(l) * 100


def fill_all_df(groupby):

    start_vesting_unique = groupby.start_vesting.unique().values
    end_vesting_unique = groupby.end_vesting.unique().values
    amount_sum = groupby.amount.sum().values

    # convert to timesamp
    start_timestamps = [
        pd.Timestamp(start[0]) for start in start_vesting_unique
    ]
    end_timestamps = [pd.Timestamp(end[0]) for end in end_vesting_unique]

    start_abs = min(start_timestamps)
    end_abs = max(end_timestamps)
    num_days_abs = (end_abs - start_abs).days + 1
    index_abs = pd.date_range(start_abs, end_abs, freq='D')

    token_list = []

    for start, end, amount in zip(
        start_timestamps, end_timestamps, amount_sum
    ):

        days_prior = (start - start_abs).days + 1
        days_after = (end_abs - end).days + 1

        num_days = (end - start).days + 1
        tokens = np.linspace(0, amount, num_days)

        tokens_after = np.full(days_after, tokens[-1])
        tokens_prior = np.zeros(days_prior)

        total_tokens = np.concatenate((tokens_prior, tokens, tokens_after))[
            :num_days_abs
        ].copy()

        token_list.append(total_tokens)

    return token_list, index_abs, start_abs, end_abs


def fill_df(groupby, index_abs, start_abs, end_abs):

    start_vesting_unique = groupby.start_vesting.unique().values
    end_vesting_unique = groupby.end_vesting.unique().values
    amount_sum = groupby.amount.sum().values

    # convert to timesamp
    start_timestamps = [
        pd.Timestamp(start[0]) for start in start_vesting_unique
    ]
    end_timestamps = [pd.Timestamp(end[0]) for end in end_vesting_unique]

    num_days_abs = (end_abs - start_abs).days + 1

    token_list = []

    for start, end, amount in zip(
        start_timestamps, end_timestamps, amount_sum
    ):

        days_prior = (start - start_abs).days + 1
        days_after = (end_abs - end).days + 1

        num_days = (end - start).days + 1
        if num_days == 1:
            tokens = np.array([amount])
        else:
            tokens = np.linspace(0, amount, num_days)
        tokens = np.linspace(0, amount, num_days)

        tokens_after = np.full(days_after, tokens[-1])
        tokens_prior = np.zeros(days_prior)

        total_tokens = np.concatenate((tokens_prior, tokens, tokens_after))[
            :num_days_abs
        ].copy()

        token_list.append(total_tokens)

    return token_list


def distribution_pie(labels, values, title):

    # plotly pie chart of final token distribution
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
            )
        ]
    )
    fig.update_traces(hoverinfo='label+percent', textinfo='label+percent')
    fig.update_layout(title_text=f'{title}', title_x=0.45)

    return fig


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_data():

    # etherscan
    etherscan_key = 'AAY3AB3K9VFEBBVP5WUSBG4TYQWGJGF14R'
    etherscan = Etherscan('AAY3AB3K9VFEBBVP5WUSBG4TYQWGJGF14R')

    # alchemy URL
    alchemy_url = (
        'https://eth-mainnet.g.alchemy.com/v2/EYt6FzjS66jI581m5jZp4izGPBoYdmlo'
    )
    w3 = Web3(Web3.HTTPProvider(alchemy_url))

    # =============================================
    # EXPECTED ALLOCATION DATA (LDO)
    # =============================================

    start_og = pd.Timestamp('2021-12-18')
    end_og = pd.Timestamp('2022-12-18')
    date_range_og = pd.date_range(start_og, end_og, freq='D')

    total_tokens_og = 1e9
    dao_tokens = np.full(len(date_range_og), 0.3632e9)
    investors_og_tokens = np.linspace(0, 0.2218e9, len(date_range_og))
    team_validators_tokens = np.linspace(0, 0.415e9, len(date_range_og))

    lido_og = pd.DataFrame(
        [dao_tokens, team_validators_tokens, investors_og_tokens],
        columns=date_range_og,
    ).T.rename(columns={0: 'DAO', 1: 'Team & Validators', 2: 'Investors'})

    # =============================================
    # ACTUAL ALLOCATION DATA (LDO)
    # =============================================

    # -----------------------------------
    # get vesting data from Vesting Contract

    lido_vesting = (
        '0xf73a1260d222f447210581DDf212D915c09a3249'  # proxy contract
    )
    aragon_token_manager = '0xde3A93028F2283cc28756B3674BD657eaFB992f4'  # og contract used for vesting

    checksum = Web3.toChecksumAddress(lido_vesting)   # checksum address
    lido_abi = etherscan.get_contract_abi(
        aragon_token_manager
    )   # get abi from original aragon contract

    # create VestingContract instance
    lido_vesting_instance = VestingContract(
        address=checksum,
        etherscan_key=etherscan_key,
        web3_url=alchemy_url,
        abi=lido_abi,
    )

    # call all new vesting events from vesting contract
    new_vesting_events = lido_vesting_instance.get_all_events('NewVesting')

    # addresses of all vesting recipients
    vesting_recipients = [e['args']['receiver'] for e in new_vesting_events]

    # initialise getVesting function from vesting contract
    getVesting_function = lido_vesting_instance.function_by_signature(
        'getVesting(address,uint256)'
    )

    dec = 10**18

    recipient_cheksum = []
    amount = []
    start_vesting = []
    cliff = []
    end_cliff = []

    # get vesting data for each recipient
    for recipient in vesting_recipients:

        check_sum = w3.toChecksumAddress(recipient)
        get_vesting = getVesting_function(check_sum, 0).call()

        recipient_cheksum.append(check_sum)
        amount.append(get_vesting[0] // dec)
        start_vesting.append(datetime.datetime.fromtimestamp(get_vesting[1]))
        cliff.append(datetime.datetime.fromtimestamp(get_vesting[2]))
        end_cliff.append(datetime.datetime.fromtimestamp(get_vesting[3]))

    # vesting data dataframe
    ldo_vesting_data = pd.DataFrame(
        [recipient_cheksum, amount, start_vesting, cliff, end_cliff]
    ).T.rename(
        columns={
            0: 'recipient',
            1: 'amount',
            2: 'start_vesting',
            3: 'cliff',
            4: 'end_vesting',
        }
    )

    # vesting team addresses
    team_addresses = pd.read_csv(
        'https://raw.githubusercontent.com/kodi1453/jarvis/main/data/vesting_addresses.csv'
    )

    # -----------------------------------
    # First Cohort (dao, team, validators, first investors)
    first_cohort = ldo_vesting_data[
        ldo_vesting_data.start_vesting < '2022-01-01'
    ]

    first_cohort['recipient'] = first_cohort['recipient'].str.upper()
    team_addresses['address'] = team_addresses['address'].str.upper()

    first_cohort['entity'] = np.where(
        first_cohort['recipient'].isin(list(team_addresses.address.values)),
        'team',
        'investors',
    )
    # group by entity and sum amount
    f = first_cohort.groupby('entity').sum()

    # -----------------------------------
    # Second Cohort (second round investors)

    second_cohort = ldo_vesting_data[
        (datetime.datetime(2023, 1, 1) > ldo_vesting_data.start_vesting)
        & (ldo_vesting_data.start_vesting > datetime.datetime(2022, 1, 1))
    ]
    second_cohort['entity'] = 'investors'

    # -----------------------------------
    # Third Cohort (third round investors)

    third_cohort = ldo_vesting_data[
        ldo_vesting_data.start_vesting > '2023-01-01'
    ]
    third_cohort['entity'] = 'investors'

    first_investors_amount = f[f.index == 'investors'].amount.values[0]
    team_validators_amount = f[f.index == 'team'].amount.values[0]
    second_investors_amount = second_cohort.amount.sum()
    third_investors_amount = third_cohort.amount.sum()

    # -----------------------------------
    # Group By

    # groupby vested tokens by start and end date
    all_token_list, index_abs, start_abs, end_abs = fill_all_df(
        ldo_vesting_data.groupby(['start_vesting', 'end_vesting'])
    )

    first_cohort_investors = first_cohort[first_cohort.entity == 'investors']
    team_validators = first_cohort[first_cohort.entity == 'team']

    first_investors_token_list = fill_df(
        first_cohort_investors.groupby(['start_vesting', 'end_vesting']),
        index_abs,
        start_abs,
        end_abs,
    )
    second_investors_token_list = fill_df(
        second_cohort.groupby(['start_vesting', 'end_vesting']),
        index_abs,
        start_abs,
        end_abs,
    )
    third_investors_token_list = fill_df(
        third_cohort.groupby(['start_vesting', 'end_vesting']),
        index_abs,
        start_abs,
        end_abs,
    )
    team_validators_token_list = fill_df(
        team_validators.groupby(['start_vesting', 'end_vesting']),
        index_abs,
        start_abs,
        end_abs,
    )

    # -----------------------------------
    # DAO Treasury (LDO balances through time)

    lido_treasury_balance = (
        pd.read_csv(
            'https://raw.githubusercontent.com/kodi1453/jarvis/main/data/lido_treasury.csv'
        )
        .astype({'amount': 'float', 'date': 'datetime64[ns]'})
        .set_index('date')
    )
    # lido_treasury_balance.index = pd.to_datetime(lido_treasury_balance.index)
    lido_treasury_balance.amount = lido_treasury_balance.amount / 10**12

    dao_actual = (
        1e9
        - second_investors_amount
        - third_investors_amount
        - team_validators_amount
        - first_investors_amount
    )
    actual_token_distro = [
        dao_actual,
        team_validators_amount,
        first_investors_amount,
        second_investors_amount,
        third_investors_amount,
    ]

    return (
        lido_og,
        lido_treasury_balance,
        actual_token_distro,
        all_token_list,
        first_investors_token_list,
        second_investors_token_list,
        third_investors_token_list,
        team_validators_token_list,
        dao_actual,
        index_abs,
        first_cohort,
        second_cohort,
        third_cohort,
    )


def main():

    (
        lido_og,
        lido_treasury_balance,
        actual_token_distro,
        all_token_list,
        first_investors_token_list,
        second_investors_token_list,
        third_investors_token_list,
        team_validators_token_list,
        dao_actual,
        index_abs,
        first_cohort,
        second_cohort,
        third_cohort,
    ) = get_data()

    option = st.selectbox(
        'What kinda chart do you want to see?',
        ('OGs', 'Money Supply'),
    )

    if option == 'OGs':
        
        col1, col2 = st.columns(2)

        # -----------------------------------
        # PIE CHARTS
        # -----------------------------------

        # -----------------------------------
        # Expected

        col1.header('Expected LDO Distribution')

        # plotly pie chart of final token distribution
        fig = distribution_pie(
            labels=['DAO', 'Team & Validators', 'Investors'],
            values=lido_og.iloc[-1].values,
            title='',
        )
        col1.plotly_chart(fig, use_container_width=True)

        # -----------------------------------
        # Actual

        col2.header('Actual LDO Distribution')

        fig = distribution_pie(
            labels=[
                'DAO Treasury',
                'Team & Validators',
                'First Round Investors',
                'Second Round Investors',
                'Third Round Investors',
            ],
            values=actual_token_distro,
            title='',
        )
        col2.plotly_chart(fig, use_container_width=True)

        # -----------------------------------
        # STACKED AREA CHARTS
        # -----------------------------------

        # -----------------------------------
        # Expected

        fig = px.area(lido_og)
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Token Supply',
            template='plotly_white',
            legend_title_text='',
        )

        col1.plotly_chart(fig, use_container_width=True)

        # -----------------------------------
        # Actual

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=index_abs,
                y=sum(first_investors_token_list),
                name='First Round Investors',
                stackgroup='one',
            )
        )
        fig.add_trace(
            go.Scatter(
                x=index_abs,
                y=sum(second_investors_token_list),
                name='Second Round Investors',
                stackgroup='one',
            )
        )
        fig.add_trace(
            go.Scatter(
                x=index_abs,
                y=sum(third_investors_token_list),
                name='Third Round Investors',
                stackgroup='one',
            )
        )
        fig.add_trace(
            go.Scatter(
                x=index_abs,
                y=sum(team_validators_token_list),
                name='Team & Validators',
                stackgroup='one',
            )
        )
        fig.add_trace(
            go.Scatter(
                x=index_abs,
                y=np.full(len(index_abs), dao_actual),
                name='DAO Treasury',
                stackgroup='one',
            )
        )
        fig.update_layout(
            template='plotly_white',
        )

        col2.plotly_chart(fig, use_container_width=True)

        # -----------------------------------
        # Stacked Fraction

        # -----------------------------------
        # Expected

        fig = px.area(
            lido_og,
            groupnorm='fraction',
        )
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Token Supply',
            template='plotly_white',
            legend_title_text='',
        )
        col1.plotly_chart(fig, use_container_width=True)

        # -----------------------------------
        # Actual

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=index_abs,
                y=sum(first_investors_token_list),
                name='First Round Investors',
                stackgroup='one',
                groupnorm='fraction',
            )
        )
        fig.add_trace(
            go.Scatter(
                x=index_abs,
                y=sum(second_investors_token_list),
                name='Second Round Investors',
                stackgroup='one',
                groupnorm='fraction',
            )
        )
        fig.add_trace(
            go.Scatter(
                x=index_abs,
                y=sum(third_investors_token_list),
                name='Third Round Investors',
                stackgroup='one',
                groupnorm='fraction',
            )
        )
        fig.add_trace(
            go.Scatter(
                x=index_abs,
                y=sum(team_validators_token_list),
                name='Team & Validators',
                stackgroup='one',
                groupnorm='fraction',
            )
        )
        fig.add_trace(
            go.Scatter(
                x=index_abs,
                y=np.full(len(index_abs), dao_actual),
                name='DAO Treasury',
                stackgroup='one',
                groupnorm='fraction',
            )
        )
        fig.update_layout(
            template='plotly_white',
        )

        col2.plotly_chart(fig, use_container_width=True)

        # -----------------------------------
        # INFLATION CHARTS
        # -----------------------------------

        # -----------------------------------
        # Expected

        lido_og['total'] = lido_og.sum(axis=1)
        lido_og['new_supply'] = lido_og.total.diff()
        lido_og.dropna(inplace=True)
        lido_og['inflation'] = inflation(lido_og)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=lido_og.index,
                y=lido_og['new_supply'],
                name=f'New Monthly Supply',
                yaxis='y',
                marker=dict(line=dict(width=1.5, color='dodgerblue')),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=lido_og.index,
                y=lido_og['inflation'],
                name=f'Annual Inflation',
                yaxis='y2',
            )
        )
        fig.update_xaxes(tickangle=45)
        fig.update_layout(
            yaxis=dict(
                title=f'New Daily Supply',
                titlefont=dict(color='dodgerblue'),
                tickfont=dict(color='dodgerblue'),
                autotypenumbers='convert types',
            ),
            yaxis2=dict(
                title=f'Annual Inflation [%]',
                titlefont=dict(color='#d62728'),
                tickfont=dict(color='#d62728'),
                anchor='x',
                overlaying='y',
                side='right',
            ),
            showlegend=False,
            autosize=True,
            template='plotly_white',
        )
        col1.plotly_chart(fig, use_container_width=True)

        # -----------------------------------
        # Actual

        total_tokens = (
            sum(first_investors_token_list)
            + sum(second_investors_token_list)
            + sum(third_investors_token_list)
            + sum(team_validators_token_list)
            + np.full(len(index_abs), dao_actual)
        )
        total_tokens_df = pd.DataFrame(total_tokens, index=index_abs).rename(
            columns={0: 'total'}
        )
        total_tokens_df['new_supply'] = total_tokens_df.diff()
        total_tokens_df.dropna(inplace=True)
        total_tokens_df['inflation'] = inflation(total_tokens_df)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=total_tokens_df.index,
                y=total_tokens_df['new_supply'],
                name=f'New Monthly Supply',
                yaxis='y',
                marker=dict(line=dict(width=1.5, color='dodgerblue')),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=total_tokens_df.index,
                y=total_tokens_df['inflation'],
                name=f'Annual Inflation',
                yaxis='y2',
            )
        )
        fig.update_xaxes(tickangle=45)
        fig.update_layout(
            yaxis=dict(
                title=f'New Supply [%]',
                titlefont=dict(color='dodgerblue'),
                tickfont=dict(color='dodgerblue'),
                autotypenumbers='convert types',
            ),
            yaxis2=dict(
                title=f'Annual Inflation [%]',
                titlefont=dict(color='#d62728'),
                tickfont=dict(color='#d62728'),
                anchor='x',
                overlaying='y',
                side='right',
            ),
            showlegend=False,
            autosize=True,
            template='plotly_white',
        )

        col2.plotly_chart(fig, use_container_width=True)

    elif option == 'Money Supply':

        st.write(
            """
        First, some definitions. We will be using the following terms:
        
        - M0: Tokens that are fully liquid and in circulation. 
        - M1: M0 + Tokens in swap contracts and synthetic versions of the token, or tokens in smart contracts that can be retrieved by an action from the owner of those tokens. 
        - M2: M1 + Tokens locked in smart contracts for a time period; unclaimed tokens. 
        - M3: tokens that are locked and/or vesting. 
        
        """
        )

        # -----------------------------------
        # Treasury Balance
        # -----------------------------------

        st.subheader('Lido DAO Treasury Balance')

        st.write(
            'Only LDO tokens held by the DAO have been taken into account.'
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=lido_treasury_balance.index,
                y=lido_treasury_balance.amount.values,
                name='M0',
                fill='tozeroy',
                mode='none',
            )
        )

        fig.add_trace(
            go.Scatter(
                x=lido_treasury_balance.index,
                y=(
                    lido_treasury_balance.amount.max()
                    - lido_treasury_balance.amount.values
                ),
                name='M2',
                fill='tozeroy',
                mode='none',
            )
        )

        fig.update_layout(
            title={
                'text': 'Money Supply of LDO tokens in Lido Treasury',
                'x': 0.5,
                'y': 0.9,
                'xanchor': 'center',
                'yanchor': 'middle',
            },
            xaxis_title='Date',
            yaxis_title='LDO Tokens',
            showlegend=True,
            template='plotly_white',
        )

        st.subheader('Supply held by investors, validators and team')

        # -----------------------------------
        # Original Allocation
        # -----------------------------------
        
        col1, col2 = st.columns(2)

        fig = go.Figure(
            data=[
                go.Pie(
                    values=first_cohort.amount.values,
                    labels=first_cohort.recipient.values,
                    textinfo='none',
                    insidetextfont=dict(size=12, color='white'),
                    hoverinfo='value+percent+label',
                    hole=0.4,
                    showlegend=False,
                )
            ]
        )
        fig.update_layout(
            title_text=f'Original Lido Allocation by Address (Founders, Team, Validators, Early Investors)',
            title_x=0.5,
            template='plotly_white',
        )

        col1.plotly_chart(fig, use_container_width=True)

        fig = go.Figure(
            data=[
                go.Pie(
                    values=[
                        first_cohort[
                            first_cohort.end_vesting < datetime.datetime.now()
                        ].amount.sum(),
                        first_cohort[
                            first_cohort.end_vesting > datetime.datetime.now()
                        ].amount.sum(),
                    ],
                    labels=['M0 - In Circulation', 'M3 - Locked'],
                )
            ]
        )
        fig.update_layout(
            title_text=f'Original Allocation: LDO in Circulation', title_x=0.5
        )
        fig.update_traces(texttemplate='%{percent:.1%}')

        col2.plotly_chart(fig, use_container_width=True)

        # -----------------------------------
        # Second Round of Investors
        # -----------------------------------

        # plotly pie chart of final token distribution
        fig = go.Figure(
            data=[
                go.Pie(
                    values=second_cohort.amount.values,
                    labels=second_cohort.recipient.values,
                    textinfo='none',
                    insidetextfont=dict(size=12, color='white'),
                    hoverinfo='value+percent+label',
                    showlegend=False,
                    hole=0.4,
                )
            ]
        )
        fig.update_layout(
            title_text=f'Second Round of Investors: LDO Allocation',
            title_x=0.5,
        )

        col1.plotly_chart(fig, use_container_width=True)

        temp_df = second_cohort.copy()
        temp_df['vest_length'] = (
            pd.to_datetime(second_cohort['end_vesting'])
            - pd.to_datetime(second_cohort['start_vesting'])
        ).dt.days
        temp_df['days_since_startvest'] = (
            pd.Timestamp.today()
            - pd.to_datetime(second_cohort['start_vesting'])
        ).dt.days

        temp_df['vested_tokens'] = (
            temp_df['amount']
            * temp_df['days_since_startvest']
            / temp_df['vest_length']
        )

        # plotly pie chart of final token distribution
        fig = go.Figure(
            data=[
                go.Pie(
                    values=[
                        temp_df['vested_tokens'].values.sum(),
                        (
                            temp_df['amount'].values
                            - temp_df['vested_tokens'].values
                        ).sum(),
                    ],
                    labels=['M0 - In Circulation', 'M3 - Locked'],
                )
            ]
        )
        fig.update_layout(
            title_text=f'Second Round of Investors: LDO in Circulation',
            title_x=0.5,
        )
        fig.update_traces(texttemplate='%{percent:.1%}')

        col2.plotly_chart(fig, use_container_width=True)

        # -----------------------------------
        # Third Round of Investors
        # -----------------------------------

        # plotly pie chart of final token distribution
        fig = go.Figure(
            data=[
                go.Pie(
                    values=third_cohort.amount.values,
                    labels=third_cohort.recipient.values,
                    textinfo='none',
                    textfont=dict(size=12),
                    textposition='outside',
                    insidetextfont=dict(size=12, color='white'),
                    hoverinfo='value+percent+label',
                    hole=0.4,
                    showlegend=False,
                )
            ]
        )
        fig.update_layout(
            title_text=f'LDO Allocation by Address in Third Cohort',
            title_x=0.5,
        )

        col1.plotly_chart(fig, use_container_width=True)

        # plotly pie chart of final token distribution
        fig = go.Figure(
            data=[
                go.Pie(
                    values=[
                        third_cohort[
                            third_cohort.end_vesting < datetime.datetime.now()
                        ].amount.sum(),
                        third_cohort[
                            third_cohort.end_vesting > datetime.datetime.now()
                        ].amount.sum(),
                    ],
                    labels=['M0 - In Circulation', 'M3 - Locked'],
                )
            ]
        )
        fig.update_layout(
            title_text=f'Third Cohort: LDO in Circulation', title_x=0.5
        )
        fig.update_traces(texttemplate='%{percent:.1%}')

        col2.plotly_chart(fig, use_container_width=True)

    return


if __name__ == '__main__':
    st.set_page_config(
        page_title='Lido Supply Distribution Tracker', layout='wide'
    )
    main()

