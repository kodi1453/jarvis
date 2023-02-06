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

st.set_page_config(layout="wide")

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


@st.cache(suppress_st_warning=True)
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
    investors_og_tokens = np.full(len(date_range_og), 0.2218e9)
    team_validators_tokens = np.full(
        len(date_range_og),
        total_tokens_og - dao_tokens[0] - investors_og_tokens[0],
    )

    labels_og = ['DAO', 'Team & Validators', 'Investors']

    lido_og = pd.DataFrame(
        [dao_tokens, team_validators_tokens, investors_og_tokens],
        columns=date_range_og,
    ).T

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

    first_investors_amount = f[f.index == 'investors'].amount
    team_validators_amount = f[f.index == 'team'].amount
    second_investors_amount = second_cohort.amount.sum()
    third_investors_amount = third_cohort.amount.sum()
    dao_actual = 1e9 - second_investors_amount - third_investors_amount
    actual_token_distro = [
        dao_actual,
        team_validators_amount,
        first_investors_amount,
        second_investors_amount,
        third_investors_amount,
    ]

    return lido_og, actual_token_distro


def main():

    col1, col2 = st.columns(2)

    lido_og, actual_token_distro = get_data()

    with col1:
        st.header('Expected')

        # plotly pie chart of final token distribution
        fig = distribution_pie(
            labels=['DAO', 'Team & Validators', 'Investors'],
            values=lido_og.iloc[-1].values,
            title='Token Distribution',
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.header('Actual')

        fig = distribution_pie(
            labels=[
                'DAO Treasury',
                'Team & Validators',
                'First Round Investors',
                'Second Round Investors',
                'Third Round Investors',
            ],
            values=actual_token_distro,
            title='Token Distribution',
        )
        st.plotly_chart(fig, use_container_width=True)

    return


if __name__ == '__main__':
    st.set_page_config(page_title='Supply Distribution Tracker')
    main()
